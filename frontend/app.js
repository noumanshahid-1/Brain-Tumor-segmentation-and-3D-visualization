// Brain Tumor Segmentation frontend

// ---------- Element references ----------
const drop      = document.getElementById('drop');
const fileEl    = document.getElementById('file');
const pickBtn   = document.getElementById('pickBtn');
const fnameEl   = document.getElementById('fname');

const runBtn    = document.getElementById('runBtn');
const statusEl  = document.getElementById('status');
const resultEl  = document.getElementById('result');

const volEl       = document.getElementById('vol');
const volEdemaEl  = document.getElementById('vol-edema');
const volNetEl    = document.getElementById('vol-net');
const volEtEl     = document.getElementById('vol-et');
const tumorCountEl = document.getElementById('tumor-count');
const tumorComponentsList = document.getElementById('tumor-components-list');

const confMeanEl  = document.getElementById('conf-mean');
const confMaxEl   = document.getElementById('conf-max');
const confBarEl   = document.getElementById('conf-bar');

const dlMask      = document.getElementById('dlMask');
const dlMetadata  = document.getElementById('dlMetadata');
const openViz     = document.getElementById('openViz');
const viz3d       = document.getElementById('viz3d');
const load3dBtn   = document.getElementById('load3d');
const viz3dPlaceholder = document.getElementById('viz3d-placeholder');

const alphaEl   = document.getElementById('alpha');
const axialIdx  = document.getElementById('axialIdx');
const corIdx    = document.getElementById('corIdx');
const sagIdx    = document.getElementById('sagIdx');

const axialVal  = document.getElementById('axialVal');
const corVal    = document.getElementById('corVal');
const sagVal    = document.getElementById('sagVal');

const imgAxial  = document.getElementById('img-axial');
const imgCor    = document.getElementById('img-cor');
const imgSag    = document.getElementById('img-sag');

const mmCard     = document.getElementById('mm-card');
const imgMmFlair = document.getElementById('img-mm-flair');
const imgMmT1    = document.getElementById('img-mm-t1');
const imgMmT1ce  = document.getElementById('img-mm-t1ce');
const imgMmT2    = document.getElementById('img-mm-t2');

const reportList = document.getElementById('report-list');
const toast      = document.getElementById('toast');

const modal3d    = document.getElementById('modal3d');
const modal3dMsg = document.getElementById('modal3d-msg');

const API = window.location.origin;

// ---------- Global state ----------
let theFile       = null;
let currentId     = null;
let shape         = null;  // [sx, sy, sz]
let numModalities = 1;
let viz3dUrl      = null;



// ---------- Utilities ----------
function show(el){ if (el) el.classList.remove('hidden'); }
function hide(el){ if (el) el.classList.add('hidden'); }

function toastMsg(msg){
  if (!toast) return;
  toast.textContent = msg;
  toast.classList.remove('hidden');
  setTimeout(()=> toast.classList.add('hidden'), 2200);
}

async function safeJson(resp){
  const text = await resp.text();
  try{
    return JSON.parse(text);
  }catch{
    return { error: text };
  }
}

function toPct(x){
  return Math.max(0, Math.min(100, Math.round(x * 100)));
}

function debounce(fn, delay=160){
  let t;
  return (...args)=>{
    clearTimeout(t);
    t = setTimeout(()=> fn(...args), delay);
  };
}

// ---------- Slice URL helper ----------
function sliceURL(plane, idx, mod){
  const a = alphaEl ? parseFloat(alphaEl.value || "0.5") : 0.5;
  let url = `${API}/slice/${currentId}?plane=${plane}&idx=${idx}&alpha=${a.toFixed(2)}`;
  if (mod){
    url += `&mod=${encodeURIComponent(mod)}`;
  }
  return url;
}

// ---------- Slider bounds ----------
function setSliceSliderBounds(){
  if (!shape) return;
  const [sx, sy, sz] = shape;

  if (axialIdx){
    axialIdx.max = String(Math.max(0, sz - 1));
    axialIdx.value = String(Math.floor(sz / 2));
    if (axialVal) axialVal.textContent = axialIdx.value;
  }
  if (corIdx){
    corIdx.max = String(Math.max(0, sy - 1));
    corIdx.value = String(Math.floor(sy / 2));
    if (corVal) corVal.textContent = corIdx.value;
  }
  if (sagIdx){
    sagIdx.max = String(Math.max(0, sx - 1));
    sagIdx.value = String(Math.floor(sx / 2));
    if (sagVal) sagVal.textContent = sagIdx.value;
  }
}

// ---------- Draw slices (overlay + multimodal) ----------
function refreshSlices(){
  if (!currentId || !shape) return;

  const [sx, sy, sz] = shape;

  const axial = axialIdx ? Number(axialIdx.value) : Math.floor(sz / 2);
  const cor   = corIdx   ? Number(corIdx.value)   : Math.floor(sy / 2);
  const sag   = sagIdx   ? Number(sagIdx.value)   : Math.floor(sx / 2);

  const clamp = (v, max) => Math.max(0, Math.min(max, v));

  const axialIdxClamped = clamp(axial, sz - 1);
  const corIdxClamped   = clamp(cor,   sy - 1);
  const sagIdxClamped   = clamp(sag,   sx - 1);

  if (axialIdx) axialIdx.value = axialIdxClamped;
  if (corIdx)   corIdx.value   = corIdxClamped;
  if (sagIdx)   sagIdx.value   = sagIdxClamped;

  if (axialVal) axialVal.textContent = axialIdxClamped;
  if (corVal)   corVal.textContent   = corIdxClamped;
  if (sagVal)   sagVal.textContent   = sagIdxClamped;

  if (imgAxial) imgAxial.src = sliceURL('axial',    axialIdxClamped);
  if (imgCor)   imgCor.src   = sliceURL('coronal',  corIdxClamped);
  if (imgSag)   imgSag.src   = sliceURL('sagittal', sagIdxClamped);

  if (!mmCard || mmCard.classList.contains('hidden') || numModalities < 4) return;

  if (imgMmFlair) imgMmFlair.src = sliceURL('axial', axialIdxClamped, 'flair');
  if (imgMmT1)    imgMmT1.src    = sliceURL('axial', axialIdxClamped, 't1');
  if (imgMmT1ce)  imgMmT1ce.src  = sliceURL('axial', axialIdxClamped, 't1ce');
  if (imgMmT2)    imgMmT2.src    = sliceURL('axial', axialIdxClamped, 't2');
}

// ---------- Debounced handlers ----------
const debouncedAlpha = debounce(()=> refreshSlices(), 160);

const debouncedAxial = debounce(()=>{
  if (!axialIdx || !axialVal) return;
  axialVal.textContent = axialIdx.value;
  refreshSlices();
}, 120);

const debouncedCor = debounce(()=>{
  if (!corIdx || !corVal) return;
  corVal.textContent = corIdx.value;
  refreshSlices();
}, 120);

const debouncedSag = debounce(()=>{
  if (!sagIdx || !sagVal) return;
  sagVal.textContent = sagIdx.value;
  refreshSlices();
}, 120);

// ---------- File handling ----------
function handleFile(files){
  if (!files || !files.length) return;
  const f = files[0];
  theFile = f;
  if (fnameEl){
    fnameEl.textContent = `${f.name} (${(f.size/1024/1024).toFixed(1)} MB)`;
  }
}

if (fileEl){
  fileEl.addEventListener('change', e => handleFile(e.target.files));
}
if (pickBtn && fileEl){
  pickBtn.addEventListener('click', ()=> fileEl.click());
}
if (drop){
  drop.addEventListener('dragenter', e => { e.preventDefault(); drop.classList.add('drag'); });
  drop.addEventListener('dragover',  e => { e.preventDefault(); drop.classList.add('drag'); });
  drop.addEventListener('dragleave', e => { e.preventDefault(); drop.classList.remove('drag'); });
  drop.addEventListener('drop', e => {
    e.preventDefault();
    drop.classList.remove('drag');
    if (e.dataTransfer && e.dataTransfer.files){
      handleFile(e.dataTransfer.files);
    }
  });
}

// ---------- Tumor components renderer ----------
function fmtNum(x, digits=2){
  return (typeof x === 'number' && Number.isFinite(x)) ? x.toFixed(digits) : '–';
}

function regionLabel(key){
  const map = {
    enhancing_tumor: 'Enhancing tumor (ET)',
    edema: 'Edema (ED)',
    non_enhancing_core: 'Non-enhancing core (NET)'
  };
  return map[key] || String(key || 'Unknown').replaceAll('_', ' ');
}

function regionClass(key){
  if (key === 'enhancing_tumor') return 'region-et';
  if (key === 'edema') return 'region-edema';
  if (key === 'non_enhancing_core') return 'region-net';
  return 'region-generic';
}

function renderTumorComponents(components){
  if (!tumorComponentsList) return;
  tumorComponentsList.innerHTML = '';

  if (!components || !components.length){
    const empty = document.createElement('div');
    empty.className = 'component-empty';
    empty.textContent = 'No reportable tumor focus metadata available for this case.';
    tumorComponentsList.appendChild(empty);
    return;
  }

  components.forEach((comp, index) => {
    const card = document.createElement('article');
    card.className = index === 0
      ? 'tumor-component-card primary-focus-card'
      : 'tumor-component-card';

    const dominant = comp.dominant_region || 'unknown';
    const slices = comp.representative_slices || {};
    const center = comp.center_voxel || [];
    const bbox = comp.bbox_voxel || {};
    const bmin = bbox.min || [];
    const bmax = bbox.max || [];
    const composition = comp.composition || {};

    const compRows = Object.values(composition).map(item => `
      <div class="composition-row">
        <span>${item.name || regionLabel(item.key)}</span>
        <span>${fmtNum(item.volume_ml)} mL · ${fmtNum(item.percent, 1)}%</span>
      </div>
    `).join('');

    const contextPreviewUrl = currentId && comp.id
      ? `${API}/component_slice/${currentId}/${comp.id}?plane=axial&view=context`
      : '';

    const maskPreviewUrl = currentId && comp.id
      ? `${API}/component_slice/${currentId}/${comp.id}?plane=axial&view=mask`
      : '';

    const titleLabel = index === 0
      ? `Primary Tumor Focus · ${comp.name || `Focus ${comp.id || ''}`}`
      : comp.name || `Tumor focus ${comp.id || ''}`;

    card.innerHTML = `
      <div class="component-card-header">
        <div>
          <div class="focus-kicker">${index === 0 ? 'PRIMARY REPORTABLE FOCUS' : 'REPORTABLE FOCUS'}</div>
          <h4>${titleLabel}</h4>
          <p>${comp.description || 'Separate connected tumor focus detected in the 3D segmentation mask.'}</p>
        </div>

        <span class="region-pill ${regionClass(dominant)}">${regionLabel(dominant)}</span>
      </div>

      <div class="component-class-legend component-class-legend-top">
        <span class="legend-chip legend-chip-et">
          <i></i>
          <b>ET</b>
          <em>Enhancing tumor</em>
        </span>

        <span class="legend-chip legend-chip-ed">
          <i></i>
          <b>ED</b>
          <em>Edema</em>
        </span>

        <span class="legend-chip legend-chip-net">
          <i></i>
          <b>NET</b>
          <em>Non-enhancing core</em>
        </span>
      </div>

      <div class="component-preview-grid">
        <div class="component-preview-block">
          ${contextPreviewUrl ? `<img class="component-preview" src="${contextPreviewUrl}" alt="${comp.name || 'Tumor focus'} anatomical context preview" loading="lazy">` : ''}
          <span>Dark MRI context</span>
        </div>

        <div class="component-preview-block">
          ${maskPreviewUrl ? `<img class="component-preview" src="${maskPreviewUrl}" alt="${comp.name || 'Tumor focus'} isolated segmentation mask" loading="lazy">` : ''}
          <span>Selected mask</span>
        </div>
      </div>

      <div class="component-metrics">
        <div><strong>${fmtNum(comp.volume_ml)}</strong><span>mL</span></div>
        <div><strong>${fmtNum(comp.mean_confidence)}</strong><span>mean conf.</span></div>
        <div><strong>${comp.voxel_count ?? '–'}</strong><span>voxels</span></div>
      </div>

      <div class="composition-box">
        ${compRows || '<div class="composition-row"><span>Composition</span><span>–</span></div>'}
      </div>

      <div class="component-meta">
        <span>Center voxel: [${center.join(', ') || '–'}]</span>
        <span>Best slices: A ${slices.axial ?? '–'} · C ${slices.coronal ?? '–'} · S ${slices.sagittal ?? '–'}</span>
        <span>BBox: [${bmin.join(', ') || '–'}] → [${bmax.join(', ') || '–'}]</span>
      </div>
    `;

    tumorComponentsList.appendChild(card);
  });
}
// ---------- Predict ----------
async function predict(){
  if (!theFile){
    toastMsg('Please select a NIfTI file first.');
    return;
  }

  try{
    if (runBtn) runBtn.disabled = true;

    if (statusEl){
      statusEl.innerHTML = '<span class="spinner"></span> Running segmentation...';
      show(statusEl);
    }

    hide(resultEl);

    viz3dUrl = null;
    if (viz3d){
      viz3d.src = '';
      viz3d.classList.add('hidden');
    }
    if (viz3dPlaceholder){
      viz3dPlaceholder.classList.remove('hidden');
      const hintEl = viz3dPlaceholder.querySelector('.hint');
      if (hintEl){
        hintEl.textContent = '3D preview will be enabled after segmentation.';
      }
    }
    if (load3dBtn){
      load3dBtn.disabled = true;
      load3dBtn.textContent = 'Load 3D Preview';
    }

    shape = null;
    numModalities = 1;

    const fd = new FormData();
    fd.append('file', theFile);

    const resp = await fetch(`${API}/predict`, {
      method: 'POST',
      body: fd
    });
    const j = await safeJson(resp);

    if (!resp.ok){
      throw new Error(j.error || 'Segmentation failed');
    }

    currentId = j.id;
    if (!currentId){
      throw new Error('No case id returned from server.');
    }

    const tumorVol = j.tumor_volume_ml;
    const rv = j.region_volumes_ml || {};

    if (volEl)      volEl.textContent      = tumorVol != null ? tumorVol.toFixed(2) : '–';
    if (volEdemaEl) volEdemaEl.textContent = rv.edema != null ? rv.edema.toFixed(2) : '–';
    if (volNetEl)   volNetEl.textContent   = rv.non_enhancing_core != null ? rv.non_enhancing_core.toFixed(2) : '–';
    if (volEtEl)    volEtEl.textContent    = rv.enhancing_tumor != null ? rv.enhancing_tumor.toFixed(2) : '–';

    if (tumorCountEl) {
      if (j.raw_tumor_count != null && j.hidden_small_component_count > 0) {
        tumorCountEl.textContent = `${j.tumor_count} shown / ${j.raw_tumor_count} raw`;
      } else {
        tumorCountEl.textContent = j.tumor_count != null ? String(j.tumor_count) : '–';
      }
    }

    renderTumorComponents(j.tumor_components || []);

    const hiddenSummary = document.getElementById('hidden-components-summary');
    if (hiddenSummary) {
      const hiddenCount = j.hidden_small_component_count || 0;
      if (hiddenCount > 0) {
        const minVol = j.min_reportable_volume_ml ?? 0.10;
        const hiddenVol = j.hidden_small_components_total_ml ?? 0;
        hiddenSummary.textContent = `${hiddenCount} sub-threshold segmentation island${hiddenCount === 1 ? '' : 's'} below ${minVol.toFixed(2)} mL were excluded from the reportable lesion list (${hiddenVol.toFixed(2)} mL total) and retained in metadata for auditability.`;
        hiddenSummary.classList.remove('hidden');
      } else {
        hiddenSummary.textContent = '';
        hiddenSummary.classList.add('hidden');
      }
    }

    const meanConf = j.mean_confidence;
    const maxConf  = j.max_confidence;

    if (confMeanEl) confMeanEl.textContent = meanConf != null ? meanConf.toFixed(2) : '–';
    if (confMaxEl)  confMaxEl.textContent  = maxConf  != null ? maxConf.toFixed(2)  : '–';

    if (confBarEl){
      const m = meanConf != null ? Math.max(0, Math.min(1, meanConf)) : 0;
      confBarEl.style.width = `${toPct(m)}%`;
      if (m >= 0.85){
        confBarEl.style.background = 'var(--success)';
      }else if (m >= 0.6){
        confBarEl.style.background = 'var(--warn)';
      }else{
        confBarEl.style.background = 'var(--danger)';
      }
    }

    if (dlMask && j.mask_url){
      dlMask.href = `${API}${j.mask_url}`;
    }
    if (openViz && j.viz3d_url){
      openViz.href = `${API}${j.viz3d_url}`;
    }
    if (dlMetadata && j.metadata_url){
      dlMetadata.href = `${API}${j.metadata_url}`;
    }

    viz3dUrl = j.viz3d_url ? `${API}${j.viz3d_url}` : null;

    if (load3dBtn){
      const has3D = !!viz3dUrl;
      load3dBtn.disabled = !has3D;
      load3dBtn.textContent = has3D ? 'Load 3D Preview' : '3D Preview Unavailable';
    }

    if (viz3dPlaceholder){
      viz3dPlaceholder.classList.remove('hidden');
      const hintEl = viz3dPlaceholder.querySelector('.hint');
      if (hintEl){
        hintEl.textContent = viz3dUrl
          ? 'Click “Load 3D Preview” to render the 3D segmentation.'
          : '3D preview is not available for this case.';
      }
    }

    const infoResp = await fetch(`${API}/case/${currentId}/info`);
    const info = await safeJson(infoResp);

    if (!infoResp.ok){
      throw new Error(info.error || 'Could not load case info');
    }

    shape         = info.shape;
    numModalities = info.num_modalities || 1;

    if (mmCard){
      if (numModalities >= 4){
        mmCard.classList.remove('hidden');
      }else{
        mmCard.classList.add('hidden');
      }
    }

    setSliceSliderBounds();
    refreshSlices();

    if (reportList){
      reportList.innerHTML = '';
      const lines = j.report_lines || [];

      if (!lines.length){
        const li = document.createElement('li');
        li.textContent = 'No summary generated for this case.';
        reportList.appendChild(li);
      }else{
        lines.forEach(line=>{
          const li = document.createElement('li');
          li.textContent = line;
          reportList.appendChild(li);
        });
      }
    }

    show(resultEl);
    toastMsg('Segmentation complete ✅');

  }catch(err){
    console.error('Error in predict():', err);
    toastMsg(err.message || 'Something went wrong.');
  }finally{
    if (statusEl) hide(statusEl);
    if (runBtn) runBtn.disabled = false;
  }
}

// ---------- Wire sliders ----------
if (alphaEl){
  alphaEl.addEventListener('input', debouncedAlpha);
}
if (axialIdx){
  axialIdx.addEventListener('input', debouncedAxial);
  axialIdx.addEventListener('change', debouncedAxial);
}
if (corIdx){
  corIdx.addEventListener('input', debouncedCor);
  corIdx.addEventListener('change', debouncedCor);
}
if (sagIdx){
  sagIdx.addEventListener('input', debouncedSag);
  sagIdx.addEventListener('change', debouncedSag);
}

// ---------- Load 3D button with modal ----------
if (load3dBtn && viz3d){
  load3dBtn.addEventListener('click', () => {
    if (!viz3dUrl) return;

    if (modal3d){
      modal3d.classList.remove('hidden');
      if (modal3dMsg){
        modal3dMsg.textContent = 'Rendering 3D visualization…';
      }
    }

    load3dBtn.disabled = true;
    load3dBtn.textContent = 'Loading...';

    if (viz3dPlaceholder){
      viz3dPlaceholder.classList.remove('hidden');
    }

    const onLoad = () => {
      load3dBtn.disabled = false;
      load3dBtn.textContent = 'Reload 3D Preview';

      if (viz3dPlaceholder){
        viz3dPlaceholder.classList.add('hidden');
      }
      if (modal3d){
        modal3d.classList.add('hidden');
      }

      viz3d.removeEventListener('load', onLoad);
    };

    viz3d.addEventListener('load', onLoad);

    const url = viz3dUrl.includes('?')
      ? `${viz3dUrl}&_t=${Date.now()}`
      : `${viz3dUrl}?_t=${Date.now()}`;

    viz3d.src = url;
    viz3d.classList.remove('hidden');
  });
}

// ---------- Run button + Enter key ----------
if (runBtn){
  runBtn.addEventListener('click', predict);
}

document.addEventListener('keydown', e => {
  if (e.key === 'Enter' && runBtn && !runBtn.disabled){
    predict();
  }
});

// ---------- Initial clean state ----------
hide(resultEl);
if (statusEl) hide(statusEl);
if (modal3d) modal3d.classList.add('hidden');