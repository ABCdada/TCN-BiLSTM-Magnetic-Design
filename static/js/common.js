(() => {
  const page = document.body.dataset.page;
  document.querySelector(`[data-nav="${page}"]`)?.classList.add('active');

  const toggle = document.querySelector('.nav-toggle');
  const nav = document.querySelector('.site-nav');
  toggle?.addEventListener('click', () => {
    const open = nav.classList.toggle('open');
    toggle.setAttribute('aria-expanded', String(open));
  });

  function messageFromError(payload, fallback) {
    if (!payload) return fallback;
    if (typeof payload.error === 'string') return payload.error;
    if (payload.error?.message) return payload.error.message;
    if (payload.message) return payload.message;
    return fallback;
  }

  async function api(url, options = {}) {
    const response = await fetch(url, options);
    const type = response.headers.get('content-type') || '';
    const payload = type.includes('application/json') ? await response.json() : null;
    if (!response.ok) throw new Error(messageFromError(payload, `${response.status} ${response.statusText}`));
    return payload;
  }

  function parseNumbers(value, preserveNull = false) {
    if (!value?.trim()) return [];
    const tokens = value.trim().split(/[\s,;]+/);
    return tokens.map((token) => {
      if (preserveNull && /^(null|nan|-)$/i.test(token)) return null;
      const number = Number(token);
      if (!Number.isFinite(number)) throw new Error(`"${token}" is not a valid number.`);
      return number;
    });
  }

  function formatMetric(value, digits = 4) {
    if (value === undefined || value === null || !Number.isFinite(Number(value))) return '-';
    const number = Number(value);
    if (Math.abs(number) >= 1000 || (Math.abs(number) > 0 && Math.abs(number) < .001)) return number.toExponential(3);
    return number.toFixed(digits).replace(/\.?0+$/, '');
  }

  function toast(message, type = 'info') {
    const region = document.querySelector('#toast-region');
    const node = document.createElement('div');
    node.className = `toast ${type}`;
    node.textContent = message;
    region.append(node);
    setTimeout(() => node.remove(), 4200);
  }

  function downloadCsv(filename, columns) {
    const header = Object.keys(columns);
    const length = Math.max(...Object.values(columns).map((values) => values.length));
    const rows = [header.join(',')];
    for (let index = 0; index < length; index += 1) rows.push(header.map((name) => columns[name][index] ?? '').join(','));
    const blob = new Blob([rows.join('\n')], { type: 'text/csv;charset=utf-8' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.click();
    URL.revokeObjectURL(link.href);
  }

  window.MagneticUI = { api, parseNumbers, formatMetric, toast, downloadCsv };
})();
