(() => {
  'use strict';

  const ui = window.MagneticUI;
  const form = document.querySelector('#circuit-form');
  const material = document.querySelector('#circuit-material');
  const modeSelect = document.querySelector('#converter-mode');
  const button = document.querySelector('#run-circuit');
  const errorBox = document.querySelector('#circuit-error');
  const charts = {};
  const palette = {
    ink: '#173a49', teal: '#087f7a', coral: '#e66d5a', amber: '#d59c36',
    blue: '#3178a8', violet: '#765a9b', grid: '#e3e9e7', muted: '#718188'
  };
  const topologyLabels = {
    single_phase_full_bridge: 'Single-phase full bridge',
    three_phase_full_bridge: 'Three-phase full bridge',
    three_phase_t_type: 'Three-phase T-type'
  };

  function setBusy(busy) {
    button.disabled = busy;
    button.classList.toggle('loading', busy);
    button.querySelector('span').textContent = busy ? 'Solving window...' : 'Run converter window';
  }

  function showError(message) {
    errorBox.textContent = message;
    errorBox.hidden = false;
    const status = document.querySelector('#circuit-status');
    status.className = 'result-status error';
    status.textContent = 'Request failed';
  }

  function numeric(id, { positive = false, nonnegative = false } = {}) {
    const value = Number(document.querySelector(`#${id}`).value);
    if (!Number.isFinite(value)) throw new Error(`${id.replaceAll('-', ' ')} must be finite.`);
    if (positive && value <= 0) throw new Error(`${id.replaceAll('-', ' ')} must be positive.`);
    if (nonnegative && value < 0) throw new Error(`${id.replaceAll('-', ' ')} cannot be negative.`);
    return value;
  }

  function buildPayload() {
    const temperature = numeric('circuit-temperature');
    if (temperature < 25 || temperature > 70) throw new Error('Temperature must remain within 25–70 °C.');
    return {
      material: material.value,
      topology: document.querySelector('#converter-topology').value,
      mode: modeSelect.value,
      temperature,
      dc_voltage: numeric('dc-voltage', { positive: true }),
      grid_voltage_rms: numeric('grid-voltage-rms', { positive: true }),
      grid_frequency_hz: numeric('grid-frequency', { positive: true }),
      switching_frequency_hz: numeric('switching-frequency', { positive: true }) * 1000,
      filter_inductance_h: numeric('filter-inductance', { positive: true }) * 1e-6,
      series_resistance_ohm: numeric('series-resistance', { nonnegative: true }),
      initial_current_A: numeric('initial-current'),
      grid_phase_deg: numeric('grid-phase'),
      turns: numeric('turns', { positive: true }),
      area_m2: numeric('area', { positive: true }) * 1e-6,
      path_length_m: numeric('path-length', { positive: true }) * 1e-3,
      initial_B: numeric('initial-b'),
      sampling_rate_hz: numeric('circuit-sampling-rate', { positive: true })
    };
  }

  function chartOptions(xTitle, yTitle, parsing = true) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      parsing,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          labels: { color: palette.muted, boxWidth: 18, usePointStyle: true, pointStyle: 'line' }
        }
      },
      scales: {
        x: {
          type: 'linear', grid: { color: palette.grid },
          ticks: { color: palette.muted, maxTicksLimit: 8 },
          title: { display: true, text: xTitle, color: palette.muted }
        },
        y: {
          grid: { color: palette.grid }, ticks: { color: palette.muted, maxTicksLimit: 6 },
          title: { display: true, text: yTitle, color: palette.muted }
        }
      }
    };
  }

  function makeChart(key, id, config) {
    charts[key]?.destroy();
    charts[key] = new Chart(document.querySelector(`#${id}`), config);
  }

  function valuesAtTime(values, timeSeconds) {
    return values.map((y, index) => ({ x: timeSeconds[index] * 1e6, y }));
  }

  function fullTarget(values, fullLength, historyLength) {
    if (!Array.isArray(values)) return Array(fullLength).fill(null);
    if (values.length === fullLength) return values.map((value, index) => (index < historyLength ? null : value));
    const target = Array(fullLength).fill(null);
    values.slice(0, fullLength - historyLength).forEach((value, index) => { target[historyLength + index] = value; });
    return target;
  }

  function fullModelCurrent(data, fullH, fullLength, historyLength, pathLength, turns) {
    const returned = data.predicted_magnetizing_current_A;
    if (Array.isArray(returned) && returned.length === fullLength) return returned;
    if (Array.isArray(fullH) && fullH.length === fullLength) return fullH.map((value) => value * pathLength / turns);
    return fullTarget(returned, fullLength, historyLength);
  }

  function metric(metrics, ...keys) {
    for (const key of keys) {
      const value = metrics?.[key];
      if (value !== undefined && value !== null) return value;
    }
    return null;
  }

  function render(data, payload) {
    const simulation = data.simulation || {};
    const time = simulation.time_s;
    const gridVoltage = simulation.grid_voltage_V;
    const inverterVoltage = simulation.inverter_voltage_V;
    const inducedVoltage = simulation.induced_voltage_V;
    const nominalCurrent = simulation.nominal_current_A;
    const b = data.B;
    const fullH = data.H;
    if (![time, gridVoltage, inverterVoltage, inducedVoltage, nominalCurrent, b, fullH].every(Array.isArray)) {
      throw new Error('The converter response is missing one or more required waveform arrays.');
    }
    const length = time.length;
    if (![gridVoltage, inverterVoltage, inducedVoltage, nominalCurrent, b, fullH].every((values) => values.length === length)) {
      throw new Error('The converter response contains inconsistent waveform lengths.');
    }
    const history = Number(payload.mode.split('-')[0]);
    const predictedTargetH = fullTarget(data.predicted_H, length, history);
    const knownH = fullH.map((value, index) => (index < history ? value : null));
    const modelCurrent = fullModelCurrent(data, fullH, length, history, payload.path_length_m, payload.turns);
    const metrics = data.metrics || {};

    const topology = simulation.topology || payload.topology;
    document.querySelector('#metric-topology').textContent = topologyLabels[topology] || topology;
    const durationMicroseconds = (time[length - 1] - time[0] + 1 / payload.sampling_rate_hz) * 1e6;
    const cycles = payload.switching_frequency_hz * durationMicroseconds * 1e-6;
    document.querySelector('#metric-window').textContent = `${ui.formatMetric(durationMicroseconds, 2)} µs / ${ui.formatMetric(cycles, 2)}`;
    document.querySelector('#circuit-energy').textContent = ui.formatMetric(metric(
      metrics,
      'predicted_target_h_db_integral_j_per_m3',
      'predicted_h_db_integral_j_per_m3'
    ));
    document.querySelector('#circuit-loss').textContent = ui.formatMetric(metric(
      metrics,
      'predicted_closed_path_energy_density_j_per_m3'
    ));

    const status = document.querySelector('#circuit-status');
    status.className = 'result-status success';
    status.textContent = 'Window complete';

    makeChart('voltage', 'voltage-chart', {
      type: 'line',
      data: { datasets: [
        { label: 'Grid voltage', data: valuesAtTime(gridVoltage, time), borderColor: palette.blue, borderWidth: 1.5, pointRadius: 0 },
        { label: 'Inverter voltage', data: valuesAtTime(inverterVoltage, time), borderColor: palette.amber, borderWidth: 1.1, pointRadius: 0 },
        { label: 'Induced voltage', data: valuesAtTime(inducedVoltage, time), borderColor: palette.ink, borderWidth: 1.4, pointRadius: 0 }
      ] },
      options: chartOptions('Time within window (µs)', 'Voltage (V)')
    });
    makeChart('current', 'current-chart', {
      type: 'line',
      data: { datasets: [
        { label: 'Nominal fixed-L current', data: valuesAtTime(nominalCurrent, time), borderColor: palette.ink, borderWidth: 1.5, pointRadius: 0 },
        { label: 'AI magnetic-branch current', data: valuesAtTime(modelCurrent, time), borderColor: palette.coral, borderWidth: 1.8, pointRadius: 0 }
      ] },
      options: chartOptions('Time within window (µs)', 'Current (A)')
    });
    makeChart('b', 'circuit-b-chart', {
      type: 'line',
      data: { datasets: [{ label: 'Reconstructed B', data: valuesAtTime(b, time), borderColor: palette.teal, borderWidth: 1.7, pointRadius: 0 }] },
      options: chartOptions('Time within window (µs)', 'B (T)')
    });
    makeChart('h', 'circuit-h-chart', {
      type: 'line',
      data: { datasets: [
        { label: 'Known H history', data: valuesAtTime(knownH, time), borderColor: palette.ink, borderWidth: 1.8, pointRadius: 0, spanGaps: false },
        { label: 'Predicted H target', data: valuesAtTime(predictedTargetH, time), borderColor: palette.coral, borderWidth: 1.9, pointRadius: 0, spanGaps: false }
      ] },
      options: chartOptions('Time within window (µs)', 'H (A/m)')
    });
    makeChart('bh', 'circuit-bh-chart', {
      type: 'line',
      data: { datasets: [{
        label: 'AI-reconstructed B-H',
        data: fullH.map((hValue, index) => ({ x: hValue, y: b[index] })),
        borderColor: palette.violet, borderWidth: 1.7, pointRadius: 0
      }] },
      options: chartOptions('H (A/m)', 'B (T)', false)
    });
  }

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    errorBox.hidden = true;
    setBusy(true);
    try {
      const payload = buildPayload();
      const data = await ui.api('/api/converter-demo', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      render(data, payload);
      ui.toast('Deterministic converter window and magnetic post-processing completed.');
    } catch (error) {
      showError(error.message);
    } finally {
      setBusy(false);
    }
  });
})();
