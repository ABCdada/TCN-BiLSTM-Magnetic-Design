(() => {
  const ui = window.MagneticUI;
  const form = document.querySelector('#prediction-form');
  const material = document.querySelector('#material');
  const mode = document.querySelector('#mode');
  const bInput = document.querySelector('#b-values');
  const hInput = document.querySelector('#h-values');
  const fileInput = document.querySelector('#csv-file');
  const errorBox = document.querySelector('#form-error');
  const button = document.querySelector('#predict-button');
  const charts = {};
  let fileData = null;
  let lastResult = null;

  const historyByMode = { '1-999': 1, '100-900': 100, '500-500': 500, '900-100': 900 };
  const palette = { ink: '#173a49', teal: '#087f7a', coral: '#e66d5a', amber: '#d59c36', grid: '#e3e9e7', muted: '#718188' };

  function setBusy(busy) {
    button.disabled = busy;
    button.classList.toggle('loading', busy);
    button.querySelector('span').textContent = busy ? 'Running inference...' : 'Run model inference';
  }

  function showError(message) {
    errorBox.textContent = message;
    errorBox.hidden = false;
    document.querySelector('#result-status').className = 'result-status error';
    document.querySelector('#result-status').textContent = 'Request failed';
  }

  function clearError() { errorBox.hidden = true; }

  function countValues(input, target) {
    try { target.textContent = ui.parseNumbers(input.value, true).length; }
    catch { target.textContent = 'invalid'; }
  }
  bInput.addEventListener('input', () => countValues(bInput, document.querySelector('#b-count')));
  hInput.addEventListener('input', () => countValues(hInput, document.querySelector('#h-count')));

  document.querySelectorAll('[data-input-tab]').forEach((tab) => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('[data-input-tab]').forEach((item) => { item.classList.remove('active'); item.setAttribute('aria-selected', 'false'); });
      document.querySelectorAll('[data-input-pane]').forEach((pane) => pane.classList.remove('active'));
      tab.classList.add('active');
      tab.setAttribute('aria-selected', 'true');
      document.querySelector(`[data-input-pane="${tab.dataset.inputTab}"]`).classList.add('active');
    });
  });

  function parseCsv(text) {
    const lines = text.replace(/^\uFEFF/, '').split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
    if (!lines.length) throw new Error('The selected file is empty.');
    const delimiter = lines[0].includes(',') ? ',' : (lines[0].includes('\t') ? '\t' : null);
    const cells = (line) => delimiter ? line.split(delimiter).map((cell) => cell.trim()) : line.split(/\s+/);
    const first = cells(lines[0]);
    // Scientific notation contains the letter "e", so header detection must
    // test numeric parseability instead of merely searching for letters.
    const header = first.some((cell) => cell === '' || !Number.isFinite(Number(cell)));
    const names = header ? first.map((cell) => cell.toLowerCase().replace(/[^a-z]/g, '')) : [];
    const rows = lines.slice(header ? 1 : 0).map(cells).filter((row) => row.some((cell) => cell !== ''));
    const bIndex = header ? Math.max(0, names.findIndex((name) => /^(b|fluxdensity)/.test(name))) : 0;
    const hFound = header ? names.findIndex((name) => /^(h|fieldstrength)/.test(name)) : (rows[0]?.length > 1 ? 1 : -1);
    const b = rows.map((row, index) => {
      const value = Number(row[bIndex]);
      if (!Number.isFinite(value)) throw new Error(`Invalid B value on CSV row ${index + 1}.`);
      return value;
    });
    const h = hFound >= 0 ? rows.map((row) => Number(row[hFound])).filter(Number.isFinite) : [];
    return { b, h };
  }

  fileInput.addEventListener('change', async () => {
    const file = fileInput.files[0];
    if (!file) return;
    try {
      fileData = parseCsv(await file.text());
      document.querySelector('#file-summary').textContent = `${file.name}: ${fileData.b.length} B values${fileData.h.length ? `, ${fileData.h.length} H values` : ''}.`;
      clearError();
    } catch (error) { fileData = null; showError(error.message); }
  });

  document.querySelector('#generate-example').addEventListener('click', () => {
    const b = Array.from({ length: 1000 }, (_, index) => {
      const phase = 2 * Math.PI * index / 160;
      return .085 * Math.sin(phase) + .012 * Math.sin(3 * phase + .4);
    });
    const h = b.map((value, index) => 180 * value + 3.8 * Math.cos(2 * Math.PI * index / 160));
    bInput.value = b.map((value) => value.toFixed(7)).join('\n');
    hInput.value = h.map((value) => value.toFixed(6)).join('\n');
    countValues(bInput, document.querySelector('#b-count'));
    countValues(hInput, document.querySelector('#h-count'));
    document.querySelector('[data-input-tab="paste"]').click();
    ui.toast('Example data generated.');
  });

  function chartOptions(xTitle, yTitle, parsing = true) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      parsing,
      interaction: { mode: 'index', intersect: false },
      plugins: { legend: { labels: { color: palette.muted, boxWidth: 18, usePointStyle: true, pointStyle: 'line' } }, tooltip: { displayColors: true } },
      scales: {
        x: { type: 'linear', grid: { color: palette.grid }, ticks: { color: palette.muted, maxTicksLimit: 8 }, title: { display: true, text: xTitle, color: palette.muted } },
        y: { grid: { color: palette.grid }, ticks: { color: palette.muted, maxTicksLimit: 6 }, title: { display: true, text: yTitle, color: palette.muted } }
      }
    };
  }

  function makeChart(key, canvasId, config) {
    charts[key]?.destroy();
    charts[key] = new Chart(document.querySelector(`#${canvasId}`), config);
  }

  function series(values, dtMicroseconds) { return values.map((y, index) => ({ x: index * dtMicroseconds, y })); }

  function render(data, b, measuredH) {
    const dt = 1e6 / data.sampling_rate_hz;
    const history = data.history_length;
    const combined = data.H;
    const known = combined.map((value, index) => index < history ? value : null);
    const predicted = combined.map((value, index) => index >= history - 1 ? value : null);
    const reference = measuredH.length === 1000 ? measuredH : null;
    const energy = data.energy || {};

    document.querySelector('#metric-material').textContent = data.grade || 'checkpoint';
    document.querySelector('#metric-window').textContent = `${history} / ${1000 - history}`;
    document.querySelector('#metric-energy').textContent = ui.formatMetric(energy.h_db_integral_j_per_m3);
    document.querySelector('#metric-loss').textContent = ui.formatMetric(energy.closed_path_energy_density_j_per_m3);
    document.querySelector('#empty-results').hidden = true;
    document.querySelector('#prediction-results').hidden = false;
    const status = document.querySelector('#result-status');
    status.className = 'result-status success';
    status.textContent = 'Inference complete';

    makeChart('b', 'b-chart', { type: 'line', data: { datasets: [{ label: 'B', data: series(b, dt), borderColor: palette.teal, borderWidth: 1.8, pointRadius: 0 }] }, options: chartOptions('Time within window (us)', 'B (T)') });
    const hSets = [
      { label: 'Known H', data: series(known, dt), borderColor: palette.ink, borderWidth: 2, pointRadius: 0, spanGaps: false },
      { label: 'Predicted H', data: series(predicted, dt), borderColor: palette.coral, borderWidth: 2, borderDash: [7, 4], pointRadius: 0, spanGaps: false }
    ];
    if (reference) hSets.unshift({ label: 'Reference H', data: series(reference, dt), borderColor: palette.amber, borderWidth: 1.4, pointRadius: 0 });
    makeChart('h', 'h-chart', { type: 'line', data: { datasets: hSets }, options: chartOptions('Time within window (us)', 'H (A/m)') });
    const bhData = combined.map((x, index) => ({ x, y: b[index] }));
    makeChart('bh', 'bh-chart', { type: 'line', data: { datasets: [{ label: 'Reconstructed B-H', data: bhData, borderColor: palette.teal, borderWidth: 1.8, pointRadius: 0 }] }, options: chartOptions('H (A/m)', 'B (T)', false) });
  }

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    clearError();
    setBusy(true);
    try {
      const selectedPane = document.querySelector('[data-input-pane].active').dataset.inputPane;
      const source = selectedPane === 'upload' ? fileData : { b: ui.parseNumbers(bInput.value), h: ui.parseNumbers(hInput.value, true).filter((value) => value !== null) };
      if (!source) throw new Error('Select and parse a CSV file first.');
      if (source.b.length !== 1000) throw new Error(`B must contain exactly 1,000 values; received ${source.b.length}.`);
      const history = historyByMode[mode.value];
      if (source.h.length && source.h.length !== history && source.h.length !== 1000) throw new Error(`Known H must contain ${history} prefix values or a complete 1,000-point reference.`);
      if (!source.h.length) throw new Error(`The selected checkpoint requires a measured ${history}-sample H history prefix.`);
      const knownH = source.h.slice(0, history);
      const payload = {
        material: material.value,
        B: source.b,
        temperature: Number(document.querySelector('#temperature').value),
        known_H: knownH,
        mode: mode.value,
        sampling_rate_hz: Number(document.querySelector('#sampling-rate').value)
      };
      const data = await ui.api('/api/predict-window', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      lastResult = { data, b: source.b, referenceH: source.h.length === 1000 ? source.h : [] };
      render(data, source.b, lastResult.referenceH);
      ui.toast('Prediction completed.');
    } catch (error) { showError(error.message); }
    finally { setBusy(false); }
  });

  document.querySelector('#download-results').addEventListener('click', () => {
    if (!lastResult) return;
    ui.downloadCsv(`prediction_${lastResult.data.material}_${lastResult.data.mode}.csv`, { B_T: lastResult.b, H_A_per_m: lastResult.data.H, reference_H_A_per_m: lastResult.referenceH });
  });
})();
