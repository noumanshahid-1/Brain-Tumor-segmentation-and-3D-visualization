import os, uuid, shutil, json
from pathlib import Path
import numpy as np, nibabel as nib, matplotlib.pyplot as plt
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

MODALITY_ORDER = ["flair", "t1", "t1ce", "t2"]


# --- paths ---
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
CKPT_PATH = MODEL_DIR / "unet3d_best.pt"
RUNS_DIR = BASE_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# --- model wrapper ---
from model.infer_wrapper import Inference3D
infer = Inference3D(str(CKPT_PATH), spacing=(1.0,1.0,1.0), roi_size=(128,128,128), num_classes=4)

# --- app ---
app = FastAPI(title="Brain Tumor Segmentation API", version="1.2")

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

# Serve frontend via backend at /ui
FRONT_DIR = (BASE_DIR / ".." / "frontend").resolve()
app.mount("/ui", StaticFiles(directory=str(FRONT_DIR), html=True), name="ui")

# CORS for local access (kept; safe)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000", "http://127.0.0.1:7860"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def save_preview_png(image_path: Path, mask_path: Path, out_png: Path):
    """Save a static 3-plane preview PNG.

    QA-safe version:
      - handles both 3D and 4D NIfTI inputs
      - uses FLAIR/channel-0 for 4D input
      - normalizes grayscale before plotting
      - avoids Matplotlib RGB clipping warnings
    """
    img = nib.load(str(image_path)).get_fdata().astype(np.float32)
    msk = nib.load(str(mask_path)).get_fdata().astype(np.float32)

    # If multi-modal, use channel 0 as anatomical preview background.
    # For BraTS-style data this is commonly FLAIR in this project.
    if img.ndim == 4:
        img = img[:, :, :, 0]

    if img.ndim != 3:
        raise ValueError(f"Expected 3D or 4D NIfTI image, got shape {img.shape}")

    if msk.ndim == 4:
        msk = np.squeeze(msk)
    if msk.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape {msk.shape}")

    mid = [s // 2 for s in img.shape]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    slices = [
        ("Axial",    img[:, :, mid[2]], msk[:, :, mid[2]]),
        ("Coronal",  img[:, mid[1], :], msk[:, mid[1], :]),
        ("Sagittal", img[mid[0], :, :], msk[mid[0], :, :]),
    ]

    # Multiclass display colors:
    # 1 = ET red, 2 = ED green, 3 = NET yellow
    from matplotlib.colors import ListedColormap, BoundaryNorm

    cmap = ListedColormap([
        (0, 0, 0, 0),          # background transparent
        (1.0, 0.25, 0.25, 1), # ET red
        (0.25, 0.9, 0.45, 1), # ED green
        (1.0, 0.84, 0.20, 1), # NET yellow
    ])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)

    for ax, (name, I, M) in zip(axes, slices):
        # Robust grayscale normalization into 0..1.
        lo, hi = np.percentile(I, [1, 99])
        I_norm = np.clip((I - lo) / (hi - lo + 1e-6), 0.0, 1.0)

        ax.imshow(I_norm, cmap="gray", vmin=0.0, vmax=1.0)

        M_int = np.rint(M).astype(np.int16)
        M_masked = np.ma.masked_where(M_int <= 0, M_int)
        ax.imshow(M_masked, cmap=cmap, norm=norm, alpha=0.65)

        ax.set_title(name)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

def _find_input_path(uid: str) -> Path | None:
    p1 = RUNS_DIR / f"{uid}.nii.gz"
    p2 = RUNS_DIR / f"{uid}.nii"
    return p1 if p1.exists() else (p2 if p2.exists() else None)

def _load_raw_volume(uid: str):
    """
    Load original NIfTI volume as-is plus inferred modality names.

    Returns:
      data: np.ndarray with shape (X,Y,Z) or (X,Y,Z,C)
      mods: list of modality names, e.g. ["flair","t1","t1ce","t2"] or ["single"]
    """
    p = _find_input_path(uid)
    if p is None:
        raise FileNotFoundError("Case not found.")

    data = nib.load(str(p)).get_fdata().astype(np.float32)

    if data.ndim == 4:
        num_mod = data.shape[3]
        if num_mod == 4:
            mods = MODALITY_ORDER
        else:
            mods = [f"ch{i}" for i in range(num_mod)]
    else:
        mods = ["single"]

    return data, mods


def _load_base_volume(uid: str) -> np.ndarray:
    """
    Backwards-compatible helper: returns a single 3D volume.
    If multi-modal, uses channel-mean as neutral context.
    """
    data, _ = _load_raw_volume(uid)
    if data.ndim == 4:
        data = data.mean(axis=3)
    return data


def _load_mask(uid: str) -> np.ndarray:
    p = RUNS_DIR / f"{uid}_mask.nii.gz"
    if not p.exists():
        raise FileNotFoundError("Mask not found. Run /predict first.")
    return nib.load(str(p)).get_fdata().astype(np.float32)

def _norm01(vol: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(vol, [1, 99])
    return np.clip((vol - lo) / (hi - lo + 1e-6), 0, 1)

def _slice_plane(vol: np.ndarray, plane: str, idx: int) -> np.ndarray:
    if plane == "axial":     return vol[:, :, idx]
    if plane == "coronal":   return vol[:, idx, :]
    if plane == "sagittal":  return vol[idx, :, :]
    raise ValueError("plane must be axial|coronal|sagittal")

def _rgba_overlay(base2d: np.ndarray, mask2d: np.ndarray, alpha: float) -> np.ndarray:
    """
    Return HxWx3 uint8 PNG array with overlay.

    Supports:
      - binary masks (0/1) -> red overlay
      - multi-class masks (0..C-1) -> different colors per class
    """
    gray = (base2d * 255.0).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)

    # If mask is essentially binary, keep old behavior
    labels = np.unique(mask2d)
    if labels.max() <= 1:
        m = (mask2d > 0.5).astype(np.float32)
        overlay = np.zeros_like(rgb)
        overlay[..., 0] = 255.0  # red
        a = (m * alpha)[..., None]
        out = rgb * (1.0 - a) + overlay * a
        return out.clip(0, 255).astype(np.uint8)

    # ----- Multi-class path -----
    m = mask2d.astype(np.int32)
    overlay = np.zeros_like(rgb, dtype=np.float32)

    # Simple color map for labels 1..3 (you can change naming later)
    # e.g. 1: enhancing tumor (red), 2: edema (green), 3: non-enhancing (yellow)
    color_map = {
        1: np.array([255.0,  80.0,  80.0]),   # bright red
        2: np.array([ 80.0, 220.0, 120.0]),   # green-ish
        3: np.array([255.0, 215.0,  80.0]),   # yellow / orange
    }
    default_color = np.array([200.0, 160.0, 255.0])  # magenta-ish for any extra class

    # Paint each label
    for c in labels:
        if c <= 0:
            continue
        col = color_map.get(int(c), default_color)
        overlay[m == c] = col

    tumor_mask = (m > 0)
    a = (tumor_mask.astype(np.float32) * alpha)[..., None]
    out = rgb * (1.0 - a) + overlay * a
    return out.clip(0, 255).astype(np.uint8)



# ---------- Component-specific preview helpers ----------
def _label_connected_components(binary_mask: np.ndarray):
    """Label separate 3D tumor components using 26-connectivity."""
    binary_mask = binary_mask.astype(bool)
    try:
        from scipy import ndimage
        structure = np.ones((3, 3, 3), dtype=np.uint8)
        labeled, count = ndimage.label(binary_mask, structure=structure)
        return labeled.astype(np.int32), int(count)
    except Exception:
        # Small dependency-free fallback.
        labeled = np.zeros(binary_mask.shape, dtype=np.int32)
        visited = np.zeros(binary_mask.shape, dtype=bool)
        sx, sy, sz = binary_mask.shape
        offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                   if not (dx == 0 and dy == 0 and dz == 0)]
        component_id = 0
        for seed in np.argwhere(binary_mask):
            x, y, z = map(int, seed)
            if visited[x, y, z]:
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


def _class_color_overlay_gray(mask2d: np.ndarray, alpha: float = 0.92) -> np.ndarray:
    """Render a segmentation-only thumbnail on a neutral gray background."""
    h, w = mask2d.shape
    rgb = np.full((h, w, 3), 74, dtype=np.float32)  # neutral gray background
    m = mask2d.astype(np.int32)
    color_map = {
        1: np.array([255.0,  80.0,  80.0]),   # ET red
        2: np.array([ 80.0, 220.0, 120.0]),   # ED green
        3: np.array([255.0, 215.0,  80.0]),   # NET yellow
    }
    for label, color in color_map.items():
        region = (m == label)
        rgb[region] = rgb[region] * (1.0 - alpha) + color * alpha
    return rgb.clip(0, 255).astype(np.uint8)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        fname = file.filename
        if not fname or not (fname.endswith(".nii") or fname.endswith(".nii.gz")):
            return JSONResponse({"error": "Please upload a .nii or .nii.gz"}, status_code=400)

        # keep original extension
        uid = str(uuid.uuid4())[:8]
        ext = ".nii.gz" if fname.endswith(".nii.gz") else ".nii"
        in_path  = RUNS_DIR / f"{uid}{ext}"
        mask_out = RUNS_DIR / f"{uid}_mask.nii.gz"
        prev_out = RUNS_DIR / f"{uid}_preview.png"
        viz_out  = RUNS_DIR / f"{uid}_3d.html"

        with open(in_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # sanity read
        try:
            _ = nib.load(str(in_path))
        except Exception as e:
            return JSONResponse({"error": f"Could not read NIfTI ({in_path.name}): {str(e)}"}, status_code=400)

        stats = infer.run_file(str(in_path), str(mask_out), out_viz_html=str(viz_out))
        save_preview_png(in_path, mask_out, prev_out)

        # Persist lightweight metadata so the frontend/API can reload case details
        # without re-running inference. This does not affect existing mask/preview flow.
        meta_out = RUNS_DIR / f"{uid}_metadata.json"
        with open(meta_out, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        return {
            "id": uid,
            "tumor_volume_ml": stats["tumor_volume_ml"],
            "mean_confidence": stats["mean_confidence"],
            "max_confidence": stats["max_confidence"],
            "used_channel_fallback": stats.get("used_channel_fallback", False),
            "region_volumes_ml": stats.get("region_volumes_ml", {}),
            "class_volumes_ml": stats.get("class_volumes_ml", {}),
            "tumor_count": stats.get("tumor_count", 0),
            "raw_tumor_count": stats.get("raw_tumor_count", stats.get("tumor_count", 0)),
            "hidden_small_component_count": stats.get("hidden_small_component_count", 0),
            "hidden_small_components_total_ml": stats.get("hidden_small_components_total_ml", 0.0),
            "min_reportable_volume_ml": stats.get("min_reportable_volume_ml", 0.10),
            "tumor_components": stats.get("tumor_components", []),
            "all_tumor_components": stats.get("all_tumor_components", []),
            "legend": stats.get("legend", {}),
            # Auto-generated summary lines
            "report_lines": stats.get("report_lines", []),
            "preview_url": f"/download/{uid}/preview",
            "mask_url": f"/download/{uid}/mask",
            "metadata_url": f"/download/{uid}/metadata",
            "viz3d_url": f"/viz/{uid}"
        }

    except Exception as e:
        import traceback, sys
        traceback.print_exc(file=sys.stderr)
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/download/{uid}/{kind}")
def download(uid: str, kind: str):
    if kind == "mask":
        p = RUNS_DIR / f"{uid}_mask.nii.gz"
    elif kind == "preview":
        p = RUNS_DIR / f"{uid}_preview.png"
    elif kind == "metadata":
        p = RUNS_DIR / f"{uid}_metadata.json"
    else:
        return JSONResponse({"error": "kind must be 'mask', 'preview', or 'metadata'"}, status_code=400)
    if not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(str(p))

@app.get("/viz/{uid}")
def viz(uid: str):
    p = RUNS_DIR / f"{uid}_3d.html"
    if not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(str(p))


@app.get("/case/{uid}/metadata")
def case_metadata(uid: str):
    """Return saved segmentation metadata for a completed case."""
    p = RUNS_DIR / f"{uid}_metadata.json"
    if not p.exists():
        return JSONResponse({"error": "metadata not found. Run /predict first."}, status_code=404)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

# ====== New: slice viewer endpoints ======


@app.get("/case/{uid}/info")
def case_info(uid: str):
    """Return volume shape and modality info for slice sliders."""
    p = _find_input_path(uid)
    if p is None:
        return JSONResponse({"error": "not found"}, status_code=404)

    data, mods = _load_raw_volume(uid)
    if data.ndim == 4:
        shape = list(data.shape[:3])   # spatial shape
        num_modalities = int(data.shape[3])
    else:
        shape = list(data.shape)
        num_modalities = 1

    return {
        "id": uid,
        "shape": shape,               # [X, Y, Z]
        "num_modalities": num_modalities,
        "modalities": mods,          # e.g. ["flair","t1","t1ce","t2"] or ["single"]
    }


@app.get("/slice/{uid}")
def get_slice(
    uid: str,
    plane: str = Query("axial", pattern="^(axial|coronal|sagittal)$"),
    idx: int = 0,
    alpha: float = 0.5,
    mod: str | None = None,
):
    """Return PNG slice with overlay (grayscale + multi-class mask).

    If the case is multi-modal (e.g. BRATS: FLAIR,T1,T1ce,T2),
    the `mod` parameter selects which modality to use for the grayscale
    background. If omitted, a channel-mean background is used.
    """
    try:
        data, mods = _load_raw_volume(uid)  # data: 3D or 4D
        msk = _load_mask(uid)               # [X,Y,Z]

        # Select 3D volume for given modality
        if data.ndim == 4:
            # Multi-modal
            vol3d = None
            if mod is not None:
                mod_lower = mod.lower()
                if mod_lower in MODALITY_ORDER:
                    ch = MODALITY_ORDER.index(mod_lower)
                    if ch < data.shape[3]:
                        vol3d = data[:, :, :, ch]
            if vol3d is None:
                # Fallback: mean across channels
                vol3d = data.mean(axis=3)
        else:
            # Single-modal
            vol3d = data

        if vol3d.shape != msk.shape:
            # resample mismatch rare; bail with clear message
            return JSONResponse({"error": "Volume/mask shape mismatch."}, status_code=500)

        # clamp idx to plane size
        if plane == "axial":
            idx = max(0, min(idx, vol3d.shape[2]-1))
            base2d = _slice_plane(vol3d, plane, idx)
            mask2d = _slice_plane(msk, plane, idx)
        elif plane == "coronal":
            idx = max(0, min(idx, vol3d.shape[1]-1))
            base2d = _slice_plane(vol3d, plane, idx)
            mask2d = _slice_plane(msk, plane, idx)
        else:  # sagittal
            idx = max(0, min(idx, vol3d.shape[0]-1))
            base2d = _slice_plane(vol3d, plane, idx)
            mask2d = _slice_plane(msk, plane, idx)

        base2d = _norm01(base2d)
        png = _rgba_overlay(base2d, mask2d, alpha=float(alpha))

        bio = BytesIO()
        import PIL.Image as Image
        Image.fromarray(png).save(bio, format="PNG")
        return Response(content=bio.getvalue(), media_type="image/png")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/component_slice/{uid}/{component_id}")
def component_slice(
    uid: str,
    component_id: int,
    plane: str = Query("axial", pattern="^(axial|coronal|sagittal)$"),
    view: str = Query("context", pattern="^(context|mask)$"),
    alpha: float = 0.92,
    mod: str | None = None,
):
    """Return a professional preview for one selected tumor component.

    Cached version:
      - first request generates PNG preview
      - later requests serve cached PNG from disk
      - improves Separate Tumor Foci card loading speed

    view=context:
      - dark MRI anatomical background
      - only the selected tumor component highlighted
      - no other tumor components shown

    view=mask:
      - isolated selected tumor segmentation
      - dark neutral background
      - useful for showing what this exact component looks like
    """
    try:
        # ---------- Cache check ----------
        preview_dir = RUNS_DIR / uid / "component_previews"
        preview_dir.mkdir(parents=True, exist_ok=True)

        safe_mod = (mod or "default").replace("/", "_").replace("\\", "_")
        safe_alpha = f"{float(alpha):.2f}"
        cache_name = f"component_{component_id}_{plane}_{view}_alpha_{safe_alpha}_mod_{safe_mod}.png"
        cache_path = preview_dir / cache_name

        if cache_path.exists():
            return FileResponse(str(cache_path), media_type="image/png")

        # ---------- Load raw image and segmentation ----------
        data, mods = _load_raw_volume(uid)
        class_map = _load_mask(uid).astype(np.uint8)

        # ---------- Select anatomical background volume ----------
        # FLAIR is usually the safest default for BraTS-style MRI because
        # edema and lesion extent are usually easier to perceive.
        if data.ndim == 4:
            selected = None

            if mod is not None:
                mod_lower = mod.lower()
                if mod_lower in MODALITY_ORDER:
                    ch = MODALITY_ORDER.index(mod_lower)
                    if ch < data.shape[3]:
                        selected = data[:, :, :, ch]

            if selected is None:
                selected = data[:, :, :, 0]

            vol3d = selected
        else:
            vol3d = data

        if vol3d.shape != class_map.shape:
            return JSONResponse(
                {"error": "Volume/mask shape mismatch for component preview."},
                status_code=500,
            )

        # ---------- Rebuild connected components ----------
        labeled, count = _label_connected_components(class_map > 0)

        if component_id < 1 or component_id > count:
            return JSONResponse({"error": "component not found"}, status_code=404)

        selected_component = labeled == component_id

        if not selected_component.any():
            return JSONResponse({"error": "empty component"}, status_code=404)

        # ---------- Choose representative slice ----------
        if plane == "axial":
            counts = selected_component.sum(axis=(0, 1))
            idx = int(counts.argmax())
            base2d = vol3d[:, :, idx]
            class2d = class_map[:, :, idx]
            selected2d = selected_component[:, :, idx]

        elif plane == "coronal":
            counts = selected_component.sum(axis=(0, 2))
            idx = int(counts.argmax())
            base2d = vol3d[:, idx, :]
            class2d = class_map[:, idx, :]
            selected2d = selected_component[:, idx, :]

        else:  # sagittal
            counts = selected_component.sum(axis=(1, 2))
            idx = int(counts.argmax())
            base2d = vol3d[idx, :, :]
            class2d = class_map[idx, :, :]
            selected2d = selected_component[idx, :, :]

        if not np.any(selected2d):
            return JSONResponse(
                {"error": "component has no pixels on selected plane"},
                status_code=404,
            )

        selected_labels = class2d * selected2d.astype(np.uint8)

        # ---------- Brain-aware crop ----------
        # For context view, keep most of the brain visible.
        # For mask view, crop closer to the selected component.
        if view == "context":
            brain2d = _norm01(base2d) > 0.03
            context_mask = brain2d | selected2d
            ys_all, xs_all = np.where(context_mask)

            if xs_all.size == 0 or ys_all.size == 0:
                y0, y1 = 0, base2d.shape[0]
                x0, x1 = 0, base2d.shape[1]
            else:
                y0 = int(ys_all.min())
                y1 = int(ys_all.max()) + 1
                x0 = int(xs_all.min())
                x1 = int(xs_all.max()) + 1

                h = y1 - y0
                w = x1 - x0
                margin = int(max(18, 0.07 * max(h, w)))

                y0 = max(0, y0 - margin)
                y1 = min(base2d.shape[0], y1 + margin)
                x0 = max(0, x0 - margin)
                x1 = min(base2d.shape[1], x1 + margin)

        else:
            ys, xs = np.where(selected2d)

            y0 = max(0, int(ys.min()) - 16)
            y1 = min(selected2d.shape[0], int(ys.max()) + 17)
            x0 = max(0, int(xs.min()) - 16)
            x1 = min(selected2d.shape[1], int(xs.max()) + 17)

        labels_crop = selected_labels[y0:y1, x0:x1].astype(np.uint8)
        selected_crop = selected2d[y0:y1, x0:x1].astype(bool)

        # ---------- Color map ----------
        color_map = {
            1: np.array([255.0,  65.0,  65.0]),   # ET red
            2: np.array([ 65.0, 225.0, 120.0]),   # ED green
            3: np.array([255.0, 215.0,  45.0]),   # NET yellow
        }

        alpha = float(max(0.45, min(alpha, 0.97)))

        # ---------- Create visual ----------
        if view == "context":
            # Darker radiology background. Tumor must dominate visually.
            base_norm_full = _norm01(base2d)
            base_norm = base_norm_full[y0:y1, x0:x1]

            gray = (base_norm * 135.0 + 8.0).clip(0, 255).astype(np.uint8)
            rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)

            # Darken non-brain background even more.
            brain_crop = (_norm01(base2d) > 0.03)[y0:y1, x0:x1]
            rgb[~brain_crop] *= 0.25

        else:
            # Isolated segmentation view: intentionally clean and dark.
            h, w = labels_crop.shape
            rgb = np.zeros((h, w, 3), dtype=np.float32)
            rgb[:, :, 0] = 18.0
            rgb[:, :, 1] = 20.0
            rgb[:, :, 2] = 28.0

        # Only selected component is rendered. No other tumors are shown.
        for label, color in color_map.items():
            region = labels_crop == label
            if np.any(region):
                if view == "context":
                    rgb[region] = rgb[region] * (1.0 - alpha) + color * alpha
                else:
                    rgb[region] = color

        # ---------- Add selected-component outline ----------
        try:
            from scipy import ndimage as ndi

            dilated = ndi.binary_dilation(selected_crop, iterations=2)
            eroded = ndi.binary_erosion(selected_crop, iterations=1)
            outline = dilated & (~eroded)

            if view == "context":
                outline_color = np.array([245.0, 250.0, 255.0])
            else:
                outline_color = np.array([255.0, 255.0, 255.0])

            rgb[outline] = rgb[outline] * 0.15 + outline_color * 0.85

        except Exception:
            pass

        png = rgb.clip(0, 255).astype(np.uint8)

        # ---------- PIL canvas ----------
        import PIL.Image as Image
        import PIL.ImageDraw as ImageDraw
        import PIL.ImageFont as ImageFont

        img = Image.fromarray(png)

        resample_bilinear = Image.Resampling.BILINEAR
        resample_nearest = Image.Resampling.NEAREST

        canvas_size = 240
        inner_pad = 14
        label_band = 28

        usable_w = canvas_size - inner_pad * 2
        usable_h = canvas_size - inner_pad * 2 - label_band

        img_w, img_h = img.size
        scale = min(usable_w / max(1, img_w), usable_h / max(1, img_h))

        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))

        if view == "mask":
            img = img.resize((new_w, new_h), resample_nearest)
        else:
            img = img.resize((new_w, new_h), resample_bilinear)

        canvas = Image.new("RGB", (canvas_size, canvas_size), (10, 13, 22))

        paste_x = (canvas_size - new_w) // 2
        paste_y = inner_pad + ((usable_h - new_h) // 2)
        canvas.paste(img, (paste_x, paste_y))

        draw = ImageDraw.Draw(canvas)

        # Border
        border = (58, 70, 96) if view == "context" else (76, 86, 110)
        draw.rounded_rectangle(
            [0, 0, canvas_size - 1, canvas_size - 1],
            radius=18,
            outline=border,
            width=2,
        )

        # Labels
        plane_label = {
            "axial": "Axial",
            "coronal": "Coronal",
            "sagittal": "Sagittal",
        }.get(plane, plane.title())

        selected_classes = labels_crop[labels_crop > 0]
        dominant_label = None
        if selected_classes.size:
            vals, cnts = np.unique(selected_classes, return_counts=True)
            dominant_label = int(vals[int(cnts.argmax())])

        label_name_map = {
            1: "ET",
            2: "ED",
            3: "NET",
        }
        label_name = label_name_map[dominant_label] if dominant_label in label_name_map else "Tumor"

        title = f"{plane_label} slice {idx}" if view == "context" else "Isolated mask"

        try:
            font = ImageFont.truetype("arial.ttf", 13)
            font_bold = ImageFont.truetype("arialbd.ttf", 13)
        except Exception:
            font = ImageFont.load_default()
            font_bold = font

        band_y0 = canvas_size - label_band
        draw.rectangle([0, band_y0, canvas_size, canvas_size], fill=(7, 10, 18))

        draw.text(
            (13, band_y0 + 7),
            title,
            fill=(220, 230, 255),
            font=font,
        )

        chip_fill_map = {
            1: (255, 92, 92),
            2: (72, 220, 120),
            3: (255, 210, 54),
        }
        chip_fill = chip_fill_map[dominant_label] if dominant_label in chip_fill_map else (160, 180, 220)

        chip_text = label_name
        chip_w = max(40, 12 + len(chip_text) * 8)
        chip_h = 18
        chip_x0 = canvas_size - chip_w - 12
        chip_y0 = band_y0 + 5
        chip_x1 = canvas_size - 12
        chip_y1 = chip_y0 + chip_h

        draw.rounded_rectangle(
            [chip_x0, chip_y0, chip_x1, chip_y1],
            radius=9,
            fill=chip_fill,
        )

        draw.text(
            (chip_x0 + 7, chip_y0 + 3),
            chip_text,
            fill=(5, 8, 14),
            font=font_bold,
        )

        # ---------- Save cache + return ----------
        canvas.save(cache_path, format="PNG")
        return FileResponse(str(cache_path), media_type="image/png")

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# --- Auto-launch browser when server starts (serving UI from backend) ---
import threading, webbrowser, time
def _open_browser():
    time.sleep(3)
    webbrowser.open_new("http://127.0.0.1:7860/ui/")
threading.Thread(target=_open_browser, daemon=True).start()
