import os, torch, numpy as np, nibabel as nib
from pathlib import Path
from monai.transforms import (
    LoadImage, EnsureChannelFirst, Spacing, Orientation,
    ScaleIntensityRangePercentiles, Compose
)
from monai.inferers import sliding_window_inference
import plotly.graph_objects as go

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Inference3D:
    """
    Inference wrapper matching your training config:
      - 3D UNet, in_channels=4, out_channels=num_classes
      - same preprocessing as training
      - robust to 1-channel inputs (auto-repeat to 4)
      - writes mask.nii.gz and 3D viz HTML
    """

    def __init__(self, ckpt_path, spacing=(1.0, 1.0, 1.0), roi_size=(128, 128, 128), num_classes=4):
        from monai.networks.nets import UNet
        self.model = UNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=num_classes,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="INSTANCE",
            dropout=0.1,
        ).to(device)
        sd = torch.load(ckpt_path, map_location=device)["state_dict"]
        self.model.load_state_dict(sd)
        self.model.eval()

        self.spacing = spacing
        self.roi_size = roi_size
        self.tx = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            Spacing(pixdim=spacing, mode="bilinear"),
            Orientation(axcodes="RAS"),
            ScaleIntensityRangePercentiles(lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True),
        ])

    # ---------- helpers ----------

    @staticmethod
    def _tumor_prob_from_logits(logits: torch.Tensor):
        """
        logits: [1, C, X, Y, Z], C>=2 (0=bg, 1..=tumor classes)
        returns:
          tumor_prob: [1, X, Y, Z] = max softmax over tumor classes (1..C-1)
          probs_full: [1, C, X, Y, Z]
        """
        probs = torch.softmax(logits, dim=1)
        if probs.shape[1] > 1:
            tumor_prob = probs[:, 1:, ...].max(dim=1, keepdim=False).values
        else:
            tumor_prob = probs[:, 0]
        return tumor_prob, probs

    @staticmethod
    def _ensure_four_channels(vol: np.ndarray):
        """
        vol: (C, X, Y, Z) or (X, Y, Z)
          - if (X,Y,Z): -> (1,C,X,Y,Z) by adding channel; then repeat to 4
          - if 1ch: repeat to 4
          - if 4ch: keep
          - else: raise clear error
        returns: (vol_4c, used_fallback_bool)
        """
        if vol.ndim == 3:
            vol = vol[None, ...]  # (1, X,Y,Z)
        C = vol.shape[0]
        if C == 4:
            return vol, False
        if C == 1:
            return np.repeat(vol, 4, axis=0), True
        raise ValueError(f"Expected 1 or 4 channels, got {C}. Please upload a 4-channel NIfTI or a single-channel volume.")


    @staticmethod
    def _label_connected_components(binary_mask: np.ndarray):
        """
        Label separate 3D tumor components using 26-connectivity.

        Returns:
          labeled: uint16/int32 array with 0=background, 1..N=tumor components
          count: number of connected components

        Uses scipy.ndimage when available, with a small pure-Python fallback for
        environments where scipy is unavailable.
        """
        binary_mask = binary_mask.astype(bool)
        try:
            from scipy import ndimage
            structure = np.ones((3, 3, 3), dtype=np.uint8)  # 26-connectivity
            labeled, count = ndimage.label(binary_mask, structure=structure)
            return labeled.astype(np.int32), int(count)
        except Exception:
            # Fallback: iterative flood fill. Slower, but dependency-free.
            labeled = np.zeros(binary_mask.shape, dtype=np.int32)
            visited = np.zeros(binary_mask.shape, dtype=bool)
            sx, sy, sz = binary_mask.shape
            component_id = 0
            offsets = [
                (dx, dy, dz)
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                for dz in (-1, 0, 1)
                if not (dx == 0 and dy == 0 and dz == 0)
            ]

            seeds = np.argwhere(binary_mask & ~visited)
            for seed in seeds:
                x, y, z = map(int, seed)
                if visited[x, y, z] or not binary_mask[x, y, z]:
                    continue
                component_id += 1
                stack = [(x, y, z)]
                visited[x, y, z] = True
                labeled[x, y, z] = component_id

                while stack:
                    cx, cy, cz = stack.pop()
                    for dx, dy, dz in offsets:
                        nx, ny, nz = cx + dx, cy + dy, cz + dz
                        if nx < 0 or ny < 0 or nz < 0 or nx >= sx or ny >= sy or nz >= sz:
                            continue
                        if visited[nx, ny, nz] or not binary_mask[nx, ny, nz]:
                            continue
                        visited[nx, ny, nz] = True
                        labeled[nx, ny, nz] = component_id
                        stack.append((nx, ny, nz))

            return labeled, int(component_id)

    @staticmethod
    def _build_tumor_components(class_map: np.ndarray, tumor_prob: np.ndarray, ml_per_vox: float):
        """
        Build metadata for every separate 3D tumor component/lesion.

        A component is defined as a connected non-background region in the
        multi-class segmentation map. Each component includes per-region
        composition, centroid, bounding box, best slices, and confidence.
        """
        region_map = {
            1: "enhancing_tumor",
            2: "edema",
            3: "non_enhancing_core",
        }
        pretty = {
            "enhancing_tumor": "Enhancing tumor (ET)",
            "edema": "Edema (ED)",
            "non_enhancing_core": "Non-enhancing core (NET)",
        }

        labeled, count = Inference3D._label_connected_components(class_map > 0)
        components = []

        for cid in range(1, count + 1):
            voxels = np.argwhere(labeled == cid)
            voxel_count = int(voxels.shape[0])
            if voxel_count == 0:
                continue

            comp_mask = labeled == cid
            volume_ml = float(voxel_count * ml_per_vox)
            centroid = voxels.mean(axis=0)
            bbox_min = voxels.min(axis=0)
            bbox_max = voxels.max(axis=0)

            labels, counts = np.unique(class_map[comp_mask], return_counts=True)
            composition = {}
            dominant_label = 0
            dominant_count = 0

            for lab, cnt in zip(labels, counts):
                lab = int(lab)
                cnt = int(cnt)
                if lab <= 0:
                    continue
                key = region_map.get(lab, f"class_{lab}")
                pct = float(cnt / voxel_count * 100.0) if voxel_count else 0.0
                composition[key] = {
                    "label": lab,
                    "name": pretty.get(key, key.replace("_", " ")),
                    "voxel_count": cnt,
                    "volume_ml": float(cnt * ml_per_vox),
                    "percent": pct,
                }
                if cnt > dominant_count:
                    dominant_label = lab
                    dominant_count = cnt

            dominant_key = region_map.get(dominant_label, f"class_{dominant_label}")
            conf_vals = tumor_prob[comp_mask]
            axial_counts = comp_mask.sum(axis=(0, 1))
            coronal_counts = comp_mask.sum(axis=(0, 2))
            sagittal_counts = comp_mask.sum(axis=(1, 2))

            components.append({
                "id": int(cid),
                "name": f"Tumor focus {cid}",
                "description": (
                    f"Separate connected tumor focus, predominantly "
                    f"{pretty.get(dominant_key, dominant_key.replace('_', ' '))}."
                ),
                "voxel_count": voxel_count,
                "volume_ml": volume_ml,
                "mean_confidence": float(conf_vals.mean()) if conf_vals.size else 0.0,
                "max_confidence": float(conf_vals.max()) if conf_vals.size else 0.0,
                "dominant_region": dominant_key,
                "composition": composition,
                "center_voxel": [int(round(x)) for x in centroid.tolist()],
                "bbox_voxel": {
                    "min": [int(x) for x in bbox_min.tolist()],
                    "max": [int(x) for x in bbox_max.tolist()],
                },
                "representative_slices": {
                    "axial": int(axial_counts.argmax()) if axial_counts.size else 0,
                    "coronal": int(coronal_counts.argmax()) if coronal_counts.size else 0,
                    "sagittal": int(sagittal_counts.argmax()) if sagittal_counts.size else 0,
                },
            })

        components.sort(key=lambda item: item["volume_ml"], reverse=True)
        for rank, comp in enumerate(components, start=1):
            comp["rank_by_volume"] = rank

        return components

    @staticmethod
    def _save_3d_html(volume_norm: np.ndarray, class_map: np.ndarray, out_html: str, case_title="3D Tumor Isosurfaces"):
        """
        volume_norm: (X,Y,Z) float in [0,1]      - context volume
        class_map:   (X,Y,Z) uint8 {0,1,2,3,...} - multi-class segmentation
          0 = background
          1 = enhancing tumor (ET)
          2 = edema (ED)
          3 = non-enhancing core (NET)

        Downsamples to keep viewer responsive and renders one isosurface
        per tumor subregion with distinct colors.
        """
        V = volume_norm.astype(np.float32)
        C = class_map.astype(np.uint8)

        # Downsample factor (1 = full res, 2 = half, etc.)
        factor = 2
        Vd = V[::factor, ::factor, ::factor]
        Cd = C[::factor, ::factor, ::factor]

        X, Y, Z = Vd.shape
        gx, gy, gz = np.meshgrid(
            np.arange(X), np.arange(Y), np.arange(Z),
            indexing="ij"
        )

        fig = go.Figure()

        # ---- Context volume (brain) ----
        fig.add_trace(go.Volume(
            x=gx.flatten(), y=gy.flatten(), z=gz.flatten(),
            value=Vd.flatten(),
            opacity=0.05,
            surface_count=8,
            showscale=False,
        ))

        # ---- Per-class isosurfaces ----
        # Colors aligned with 2D overlay & legend
        class_defs = [
            (1, "Enhancing tumor (ET)",  "rgb(255, 80, 80)"),   # red
            (2, "Edema (ED)",           "rgb( 80,220,120)"),   # green
            (3, "Non-enhancing core",   "rgb(255,215, 80)"),   # yellow
        ]

        for label, name, color in class_defs:
            mask = (Cd == label).astype(np.float32)
            if mask.max() < 0.5:
                continue  # no voxels of this class after downsampling

            fig.add_trace(go.Isosurface(
                x=gx.flatten(),
                y=gy.flatten(),
                z=gz.flatten(),
                value=mask.flatten(),
                isomin=0.5,
                isomax=1.0,
                surface_count=1,
                opacity=0.6,
                showscale=False,
                colorscale=[[0.0, color], [1.0, color]],
                caps=dict(x_show=False, y_show=False, z_show=False),
                name=name,
            ))

        fig.update_layout(
            title=case_title,
            width=1000,
            height=720,
            margin=dict(l=0, r=0, t=40, b=0),
            scene=dict(
                xaxis_visible=False,
                yaxis_visible=False,
                zaxis_visible=False,
            ),
        )
        fig.write_html(out_html, include_plotlyjs="cdn")


    # ---------- main API used by backend/app.py ----------

    def run_file(self, nii_path: str, out_mask_path: str, out_viz_html: str | None = None):
        """
        Runs full inference pipeline on a single file.

        Upgraded version:
          - Saves a *multi-class* mask (0 = background, 1..C-1 = tumor subclasses)
          - Still uses a binary mask internally for stats
          - Optionally writes a 3D HTML viz
          - Returns extra per-class volume statistics
        """
        # ---------- Preprocess -> (C, X, Y, Z) ----------
        vol = self.tx(nii_path).astype(np.float32)
        vol, used_fallback = self._ensure_four_channels(vol)
        vol_t = torch.from_numpy(vol)[None].to(device)  # (1, C, X, Y, Z)

        # ---------- Model prediction ----------
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            logits = sliding_window_inference(
                vol_t, self.roi_size, sw_batch_size=1, predictor=self.model, overlap=0.5
            )

        # tumor_prob: (1, X, Y, Z)
        # probs:      (1, C, X, Y, Z)
        tumor_prob, probs = self._tumor_prob_from_logits(logits)

        # ---------- Multi-class label map ----------
        # argmax over channel dimension -> [1, X, Y, Z]
        class_map = probs.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        # Binary mask: any non-background voxel is tumor
        pred_bin = (class_map > 0).astype(np.uint8)

        # ---------- Save mask with original affine/header ----------
        src = nib.load(nii_path)
        # NOTE: This is now a SEGLABEL-like multi-class NIfTI (0..C-1)
        nib.save(nib.Nifti1Image(class_map, src.affine, src.header), out_mask_path)

        # ---------- Confidence statistics (inside tumor mask) ----------
        tp = tumor_prob[0].detach().cpu().numpy()  # (X, Y, Z)
        inside = tp[pred_bin > 0]
        if inside.size:
            mean_conf = float(inside.mean())
            max_conf = float(inside.max())
        else:
            mean_conf = 0.0
            max_conf = 0.0

        # ---------- Volume statistics (mL) ----------
        # Voxel spacing from affine (mm)
        spac = np.sqrt((src.affine[:3, :3] ** 2).sum(axis=0))
        mm3_per_vox = float(spac.prod())
        ml_per_vox = mm3_per_vox / 1000.0

        # Whole-tumor volume (any non-zero label)
        tumor_ml = float(pred_bin.sum() * ml_per_vox)

                # Per-class volumes (labels 1..C-1)
        num_classes = int(probs.shape[1])
        class_vols_ml = {}
        for c in range(1, num_classes):
            vox = int((class_map == c).sum())
            class_vols_ml[c] = float(vox * ml_per_vox)

                # Map numeric labels → biological regions
        # Consistent with 2D overlay colors and legend:
        #   1 -> Enhancing tumor (ET)      [red]
        #   2 -> Edema (ED)                [green]
        #   3 -> Non-enhancing core (NET)  [yellow]
        region_map = {
            1: "enhancing_tumor",
            2: "edema",
            3: "non_enhancing_core",
        }
        region_volumes_ml = {}
        for c, name in region_map.items():
            region_volumes_ml[name] = float(class_vols_ml.get(c, 0.0))

        # ---------- Separate tumor foci / connected components ----------
        # Keep the complete connected-component list for metadata/export, but
        # expose a professional reportable list to the UI. Very tiny isolated
        # 1-voxel / few-voxel islands are often model speckle/noise after
        # argmax segmentation, so they should not dominate the main dashboard.
        all_tumor_components = self._build_tumor_components(class_map, tp, ml_per_vox)
        min_reportable_volume_ml = 0.10
        reportable_components = [
            c for c in all_tumor_components
            if c.get("volume_ml", 0.0) >= min_reportable_volume_ml
        ]
        # Safety fallback: if the model detects only tiny components, still show
        # the largest one so the interface never looks empty.
        if not reportable_components and all_tumor_components:
            reportable_components = all_tumor_components[:1]

        small_components = [
            c for c in all_tumor_components
            if c not in reportable_components
        ]
        small_components_total_ml = float(sum(c.get("volume_ml", 0.0) for c in small_components))
        tumor_components = reportable_components

        # ---------- Simple clinical-style summary ----------
        total = float(tumor_ml)
        report_lines: list[str] = []

        pretty_name = {
            "enhancing_tumor": "Enhancing tumor (ET)",
            "edema": "Edema (ED)",
            "non_enhancing_core": "Non-enhancing core (NET)",
        }

        if total <= 0.01:
            report_lines.append(
                "No reportable tumor burden detected by the current segmentation threshold "
                "(total predicted volume below 0.01 mL)."
            )
        else:
            focus_count = len(tumor_components)
            all_focus_count = len(all_tumor_components)
            hidden_count = len(small_components)
            if hidden_count > 0:
                report_lines.append(
                    f"Detected {focus_count} reportable tumor focus{'es' if focus_count != 1 else ''} "
                    f"above {min_reportable_volume_ml:.2f} mL. "
                    f"{hidden_count} sub-threshold segmentation island{'s' if hidden_count != 1 else ''} "
                    f"below the reporting threshold were excluded from the main lesion list "
                    f"and retained in metadata for auditability."
                )
            else:
                report_lines.append(
                    f"Detected {focus_count} separate tumor focus{'es' if focus_count != 1 else ''} "
                    f"based on 3D connected-component analysis."
                )
            # Fractions per region
            fractions = {
                k: (v / total if total > 0 else 0.0)
                for k, v in region_volumes_ml.items()
            }
            # Dominant region
            if region_volumes_ml:
                dominant_key = max(region_volumes_ml, key=lambda k: region_volumes_ml[k])
                dom_pct = int(round(fractions.get(dominant_key, 0.0) * 100))
                report_lines.append(
                    f"Total tumor volume is approximately {total:.2f} mL, "
                    f"predominantly {pretty_name.get(dominant_key, dominant_key)} (~{dom_pct}%)."
                )

            et = region_volumes_ml.get("enhancing_tumor", 0.0)
            ed = region_volumes_ml.get("edema", 0.0)
            net = region_volumes_ml.get("non_enhancing_core", 0.0)

            # Extra context lines (only if > 0)
            if et > 0 or net > 0:
                report_lines.append(
                    f"Enhancing component: {et:.2f} mL; non-enhancing core: {net:.2f} mL."
                )
            if ed > 0:
                report_lines.append(
                    f"Peritumoral edema volume: {ed:.2f} mL, which may reflect local infiltration."
                )

        # ---------- 3D visualization ----------
        if out_viz_html is not None:
            v_ctx = vol.mean(axis=0)  # (X,Y,Z)
            vmin, vmax = np.percentile(v_ctx, [1, 99])
            v_norm = np.clip((v_ctx - vmin) / (vmax - vmin + 1e-6), 0, 1)
            self._save_3d_html(v_norm, class_map, out_viz_html)

        # ---------- Return metrics used by the API response ----------
        return {
            "tumor_volume_ml": tumor_ml,
            "mean_confidence": mean_conf,
            "max_confidence": max_conf,
            "used_channel_fallback": bool(used_fallback),
            "class_volumes_ml": class_vols_ml,
            "region_volumes_ml": region_volumes_ml,
            "tumor_components": tumor_components,
            "all_tumor_components": all_tumor_components,
            "tumor_count": len(tumor_components),
            "raw_tumor_count": len(all_tumor_components),
            "hidden_small_component_count": len(small_components),
            "hidden_small_components_total_ml": small_components_total_ml,
            "min_reportable_volume_ml": min_reportable_volume_ml,
            "legend": {
                "1": {"key": "enhancing_tumor", "abbr": "ET", "name": "Enhancing tumor", "color": "#f97373"},
                "2": {"key": "edema", "abbr": "ED", "name": "Peritumoral edema", "color": "#4ade80"},
                "3": {"key": "non_enhancing_core", "abbr": "NET", "name": "Non-enhancing core", "color": "#facc15"},
            },
            "report_lines": report_lines,
        }
