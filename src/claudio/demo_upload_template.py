"""
demo_upload_template.py — HTML Template for the Claudio Upload Page

Extracted from audio_demo_server.py for 300-line compliance.
Contains only the HTML/CSS/JS string used by the upload handler.
"""

from __future__ import annotations


def upload_page_html() -> str:
    """Return the full HTML for the upload & process page."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Claudio — Upload & Process</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;500;700;900&display=swap');
  :root {
    --bg: #0a0a0f; --card: #12121a; --border: #1e1e2e;
    --accent: #6366f1; --accent2: #a855f7;
    --dry: #f59e0b; --wet: #22d3ee;
    --text: #e2e8f0; --muted: #64748b;
    --success: #22c55e; --error: #ef4444;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Inter', sans-serif;
    background: var(--bg); color: var(--text);
    min-height: 100vh; padding: 2rem;
  }
  .hero { text-align: center; padding: 2rem 1rem; }
  .hero h1 {
    font-size: 2.5rem; font-weight: 900;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .hero p { color: var(--muted); margin-top: 0.5rem; }
  .nav { text-align: center; margin: 1rem 0; }
  .nav a {
    color: var(--accent); text-decoration: none; font-weight: 500;
    padding: 0.5rem 1rem; border: 1px solid var(--accent);
    border-radius: 8px; transition: all 0.2s;
  }
  .nav a:hover { background: var(--accent); color: #fff; }
  .container { max-width: 800px; margin: 0 auto; }
  .upload-zone {
    border: 2px dashed var(--border); border-radius: 20px;
    padding: 3rem 2rem; text-align: center;
    transition: all 0.3s; cursor: pointer;
    background: var(--card); margin: 2rem 0;
  }
  .upload-zone:hover, .upload-zone.dragover {
    border-color: var(--accent);
    background: rgba(99,102,241,0.08);
    box-shadow: 0 0 40px rgba(99,102,241,0.1);
  }
  .upload-zone .icon { font-size: 3rem; margin-bottom: 1rem; }
  .upload-zone h3 { font-weight: 700; margin-bottom: 0.5rem; }
  .upload-zone p { color: var(--muted); font-size: 0.9rem; }
  .upload-zone input { display: none; }
  .controls {
    display: flex; gap: 1rem; align-items: center;
    justify-content: center; flex-wrap: wrap; margin: 1.5rem 0;
  }
  .controls label { color: var(--muted); font-size: 0.85rem; font-weight: 600; }
  .controls select {
    background: var(--card); color: var(--text);
    border: 1px solid var(--border); border-radius: 8px;
    padding: 0.6rem 1rem; font-family: inherit; font-size: 0.9rem;
    cursor: pointer;
  }
  .btn {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: #fff; border: none; border-radius: 10px;
    padding: 0.8rem 2rem; font-weight: 700; font-size: 1rem;
    cursor: pointer; transition: all 0.2s; font-family: inherit;
  }
  .btn:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(99,102,241,0.4); }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
  .file-list { margin: 1rem 0; }
  .file-item {
    display: flex; align-items: center; gap: 0.8rem;
    padding: 0.6rem 1rem; background: var(--card);
    border: 1px solid var(--border); border-radius: 10px;
    margin-bottom: 0.5rem; font-size: 0.9rem;
  }
  .file-item .name { flex: 1; }
  .file-item .size { color: var(--muted); font-size: 0.8rem; }
  .file-item .remove {
    color: var(--error); cursor: pointer; font-weight: 700;
    width: 24px; height: 24px; display: flex; align-items: center;
    justify-content: center; border-radius: 50%;
    transition: background 0.2s;
  }
  .file-item .remove:hover { background: rgba(239,68,68,0.15); }
  .progress {
    background: var(--border); border-radius: 8px;
    height: 6px; margin: 1rem 0; overflow: hidden; display: none;
  }
  .progress .bar {
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    height: 100%; border-radius: 8px; width: 0%;
    transition: width 0.3s;
  }
  .results { margin: 2rem 0; }
  .result-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 14px; padding: 1.2rem 1.5rem; margin-bottom: 1rem;
    animation: fadeIn 0.4s ease;
  }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
  .result-card h4 { font-weight: 600; margin-bottom: 0.8rem; }
  .result-card .ab-row { display: flex; gap: 1rem; flex-wrap: wrap; }
  .result-card .ab-col { flex: 1; min-width: 200px; }
  .result-card .label {
    display: inline-block; font-size: 0.7rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em;
    padding: 0.2rem 0.5rem; border-radius: 5px; margin-bottom: 0.4rem;
  }
  .result-card .label.dry { background: rgba(245,158,11,0.15); color: var(--dry); }
  .result-card .label.wet { background: rgba(34,211,238,0.15); color: var(--wet); }
  .result-card audio { width: 100%; height: 36px; border-radius: 6px; }
  .result-card .meta { color: var(--muted); font-size: 0.8rem; margin-top: 0.5rem; }
  .result-card .error-msg { color: var(--error); font-size: 0.85rem; }
  .batch-info {
    text-align: center; color: var(--muted); font-size: 0.85rem;
    margin: 1rem 0; display: none;
  }
</style>
</head>
<body>
<div class="hero">
  <h1>Upload & Process</h1>
  <p>Drop your audio files, choose a position, and hear Claudio's binaural rendering</p>
</div>
<div class="nav">
  <a href="/">← Back to Demo Player</a>
</div>
<div class="container">
  <div class="upload-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
    <div class="icon">📁</div>
    <h3>Drop WAV files here</h3>
    <p>or click to browse — supports single files, multiple files, or entire folders</p>
    <input type="file" id="fileInput" accept=".wav,audio/wav" multiple webkitdirectory />
  </div>
  <div class="file-list" id="fileList"></div>
  <div class="controls">
    <label>Position:</label>
    <select id="posSelect">
      <option value="center">Center (0°)</option>
      <option value="left_30">Left 30°</option>
      <option value="left_45" selected>Left 45°</option>
      <option value="left_90">Hard Left (90°)</option>
      <option value="right_30">Right 30°</option>
      <option value="right_45">Right 45°</option>
      <option value="right_90">Hard Right (90°)</option>
      <option value="behind">Behind (180°)</option>
      <option value="above">Above</option>
    </select>
    <button class="btn" id="processBtn" disabled onclick="processFiles()">
      Process with Claudio
    </button>
  </div>
  <div class="batch-info" id="batchInfo"></div>
  <div class="progress" id="progress"><div class="bar" id="progressBar"></div></div>
  <div class="results" id="results"></div>
</div>
<script>
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const fileList = document.getElementById('fileList');
const processBtn = document.getElementById('processBtn');
const progress = document.getElementById('progress');
const progressBar = document.getElementById('progressBar');
const results = document.getElementById('results');
const batchInfo = document.getElementById('batchInfo');
let selectedFiles = [];

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('dragover');
  const items = e.dataTransfer.items;
  if (items) {
    const promises = [];
    for (const item of items) {
      const entry = item.webkitGetAsEntry && item.webkitGetAsEntry();
      if (entry) { promises.push(traverseEntry(entry)); }
      else if (item.kind === 'file') { selectedFiles.push(item.getAsFile()); }
    }
    Promise.all(promises).then(renderFileList);
  }
});

function traverseEntry(entry) {
  return new Promise(resolve => {
    if (entry.isFile) {
      entry.file(f => { if (f.name.toLowerCase().endsWith('.wav')) selectedFiles.push(f); resolve(); });
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      reader.readEntries(entries => {
        Promise.all(entries.map(traverseEntry)).then(resolve);
      });
    } else { resolve(); }
  });
}

fileInput.addEventListener('change', () => {
  for (const f of fileInput.files) {
    if (f.name.toLowerCase().endsWith('.wav')) selectedFiles.push(f);
  }
  renderFileList();
});

function renderFileList() {
  fileList.innerHTML = selectedFiles.map((f, i) => `
    <div class="file-item">
      <span>🎵</span>
      <span class="name">${f.name}</span>
      <span class="size">${(f.size / 1024).toFixed(0)} KB</span>
      <span class="remove" onclick="removeFile(${i})">✕</span>
    </div>
  `).join('');
  processBtn.disabled = selectedFiles.length === 0;
  if (selectedFiles.length > 1) {
    batchInfo.style.display = 'block';
    batchInfo.textContent = selectedFiles.length + ' files selected — will process all at selected position';
  } else { batchInfo.style.display = 'none'; }
}

function removeFile(idx) {
  selectedFiles.splice(idx, 1);
  renderFileList();
}

async function processFiles() {
  if (selectedFiles.length === 0) return;
  const pos = document.getElementById('posSelect').value;
  processBtn.disabled = true;
  progress.style.display = 'block';
  progressBar.style.width = '10%';
  results.innerHTML = '';

  const total = selectedFiles.length;
  for (let i = 0; i < total; i++) {
    progressBar.style.width = ((i / total) * 80 + 10) + '%';
    const fd = new FormData();
    fd.append('position', pos);
    fd.append('file', selectedFiles[i]);
    try {
      const resp = await fetch('/api/process', { method: 'POST', body: fd });
      const data = await resp.json();
      for (const r of data.results) {
        if (r.error) {
          results.innerHTML += `<div class="result-card"><h4>${r.filename}</h4><p class="error-msg">❌ ${r.error}</p></div>`;
        } else {
          results.innerHTML += `
            <div class="result-card">
              <h4>${r.filename} — ${r.position.replace(/_/g, ' ')}</h4>
              <div class="meta">${r.duration_s.toFixed(1)}s · ${r.sample_rate} Hz</div>
              <div class="ab-row" style="margin-top:0.8rem">
                <div class="ab-col">
                  <span class="label dry">Original</span>
                  <audio controls src="${r.original_url}"></audio>
                </div>
                <div class="ab-col">
                  <span class="label wet">Binaural (Claudio)</span>
                  <audio controls src="${r.binaural_url}"></audio>
                </div>
              </div>
            </div>`;
        }
      }
    } catch (err) {
      results.innerHTML += `<div class="result-card"><h4>${selectedFiles[i].name}</h4><p class="error-msg">❌ Network error: ${err.message}</p></div>`;
    }
  }
  progressBar.style.width = '100%';
  setTimeout(() => { progress.style.display = 'none'; progressBar.style.width = '0%'; }, 800);
  processBtn.disabled = false;
}
</script>
</body>
</html>"""
