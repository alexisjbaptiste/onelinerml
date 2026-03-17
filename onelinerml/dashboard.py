import json


def render_dashboard(metrics, analytics):
    """Render the full dashboard HTML page."""
    data_json = json.dumps({
        "metrics": metrics,
        "analytics": analytics,
    })

    return DASHBOARD_HTML.replace("__DATA_JSON__", data_json)


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OneLinerML Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0f172a; color: #e2e8f0; }
  .header { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
             padding: 24px 32px; border-bottom: 1px solid #334155; }
  .header h1 { font-size: 24px; font-weight: 700; }
  .header h1 span { color: #38bdf8; }
  .header .subtitle { color: #94a3b8; margin-top: 4px; font-size: 14px; }
  .container { max-width: 1400px; margin: 0 auto; padding: 24px; }
  .grid { display: grid; gap: 20px; }
  .grid-4 { grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); }
  .grid-2 { grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); }
  .card { background: #1e293b; border-radius: 12px; padding: 20px;
           border: 1px solid #334155; }
  .card h2 { font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em;
              color: #94a3b8; margin-bottom: 12px; }
  .metric-value { font-size: 32px; font-weight: 700; color: #38bdf8; }
  .metric-label { font-size: 13px; color: #64748b; margin-top: 4px; }
  .chart-container { position: relative; width: 100%; }
  .chart-container canvas { width: 100% !important; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; padding: 8px 12px; color: #94a3b8; border-bottom: 1px solid #334155;
       font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; font-size: 11px; }
  td { padding: 8px 12px; border-bottom: 1px solid #1e293b; }
  tr:hover td { background: #334155; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px;
           font-size: 11px; font-weight: 600; }
  .badge-blue { background: #1e3a5f; color: #38bdf8; }
  .badge-green { background: #14532d; color: #4ade80; }
  .badge-purple { background: #3b0764; color: #c084fc; }
  .section-title { font-size: 18px; font-weight: 600; margin: 32px 0 16px; }
  .api-box { background: #0f172a; border: 1px solid #334155; border-radius: 8px;
              padding: 16px; font-family: monospace; font-size: 13px; margin-top: 8px;
              color: #38bdf8; word-break: break-all; }
  .correlation-grid { display: grid; gap: 1px; background: #334155; border-radius: 8px;
                       overflow: hidden; }
  .corr-cell { padding: 8px; text-align: center; font-size: 12px; font-weight: 600; }
  .corr-label { padding: 8px; text-align: center; font-size: 11px; color: #94a3b8;
                 background: #1e293b; font-weight: 600; }
</style>
</head>
<body>
<div class="header">
  <h1><span>OneLinerML</span> Dashboard</h1>
  <div class="subtitle" id="model-subtitle"></div>
</div>
<div class="container">

  <!-- Metrics cards -->
  <div class="grid grid-4" id="metrics-cards"></div>

  <!-- Charts row -->
  <div class="section-title">Model Performance</div>
  <div class="grid grid-2">
    <div class="card">
      <h2>Predictions vs Actual</h2>
      <div class="chart-container"><canvas id="pred-chart"></canvas></div>
    </div>
    <div class="card" id="importance-card">
      <h2>Feature Importance</h2>
      <div class="chart-container"><canvas id="importance-chart"></canvas></div>
    </div>
  </div>

  <!-- Correlation + distribution -->
  <div class="section-title">Data Insights</div>
  <div class="grid grid-2">
    <div class="card" id="corr-card">
      <h2>Correlation Matrix</h2>
      <div id="corr-matrix"></div>
    </div>
    <div class="card">
      <h2>Target Distribution</h2>
      <div class="chart-container"><canvas id="dist-chart"></canvas></div>
    </div>
  </div>

  <!-- Feature stats table -->
  <div class="section-title">Feature Overview</div>
  <div class="card">
    <h2>Features</h2>
    <table id="feature-table">
      <thead><tr><th>Name</th><th>Type</th><th>Detail</th><th>Missing</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>

  <!-- API info -->
  <div class="section-title">API Endpoint</div>
  <div class="card">
    <h2>Predict</h2>
    <div class="api-box" id="api-box"></div>
  </div>

</div>

<script>
const RAW = __DATA_JSON__;
const M = RAW.metrics;
const A = RAW.analytics;

// --- Subtitle ---
document.getElementById('model-subtitle').textContent =
  `${A.dataset.model} \u2022 ${A.dataset.rows} rows \u2022 ${A.dataset.features} features \u2022 target: ${A.dataset.target}`;

// --- Metric cards ---
const cardsEl = document.getElementById('metrics-cards');
Object.entries(M).forEach(([k, v]) => {
  const card = document.createElement('div');
  card.className = 'card';
  card.innerHTML = `<div class="metric-value">${typeof v === 'number' ? v.toFixed(4) : v}</div>
                     <div class="metric-label">${k.replace(/_/g, ' ').toUpperCase()}</div>`;
  cardsEl.appendChild(card);
});

// --- Chart defaults ---
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = '#334155';
Chart.defaults.font.family = '-apple-system, BlinkMacSystemFont, sans-serif';

// --- Predictions chart ---
if (A.residuals) {
  new Chart(document.getElementById('pred-chart'), {
    type: 'scatter',
    data: { datasets: [{
      label: 'Predicted vs Actual',
      data: A.residuals.y_test.map((v, i) => ({ x: v, y: A.residuals.y_pred[i] })),
      backgroundColor: 'rgba(56, 189, 248, 0.5)',
      borderColor: '#38bdf8',
      pointRadius: 3,
    }, {
      label: 'Perfect',
      data: [
        {x: Math.min(...A.residuals.y_test), y: Math.min(...A.residuals.y_test)},
        {x: Math.max(...A.residuals.y_test), y: Math.max(...A.residuals.y_test)}
      ],
      type: 'line', borderColor: '#4ade80', borderDash: [5,5], pointRadius: 0, borderWidth: 2,
    }]},
    options: { responsive: true, plugins: { legend: { display: false } },
      scales: { x: { title: { display: true, text: 'Actual' }},
                y: { title: { display: true, text: 'Predicted' }}}}
  });
} else if (A.confusion_matrix) {
  const cm = A.confusion_matrix;
  const el = document.getElementById('pred-chart');
  const labels = cm.labels;
  new Chart(el, {
    type: 'bar',
    data: { labels: labels,
      datasets: labels.map((l, i) => ({
        label: `Predicted ${l}`,
        data: cm.values.map(row => row[i]),
        backgroundColor: `hsla(${i * 360 / labels.length}, 70%, 60%, 0.7)`,
      }))},
    options: { responsive: true, scales: { x: { stacked: true }, y: { stacked: true,
      title: { display: true, text: 'Count' }}},
      plugins: { title: { display: true, text: 'Confusion Matrix' }}}
  });
}

// --- Feature importance ---
if (A.feature_importance) {
  const fi = A.feature_importance;
  const topN = fi.names.slice(0, 15);
  const topV = fi.values.slice(0, 15);
  new Chart(document.getElementById('importance-chart'), {
    type: 'bar',
    data: { labels: topN,
      datasets: [{ data: topV, backgroundColor: '#8b5cf6', borderRadius: 4 }]},
    options: { indexAxis: 'y', responsive: true,
      plugins: { legend: { display: false } },
      scales: { x: { title: { display: true, text: 'Importance' }}}}
  });
} else {
  document.getElementById('importance-card').innerHTML =
    '<h2>Feature Importance</h2><p style="color:#64748b;padding:20px;">Not available for this model type.</p>';
}

// --- Correlation matrix ---
if (A.correlation) {
  const c = A.correlation;
  const n = c.columns.length;
  const grid = document.getElementById('corr-matrix');
  grid.style.gridTemplateColumns = `80px repeat(${n}, 1fr)`;
  grid.className = 'correlation-grid';
  // header row
  grid.innerHTML = '<div class="corr-label"></div>' +
    c.columns.map(col => `<div class="corr-label">${col.length > 8 ? col.slice(0,7) + '\u2026' : col}</div>`).join('');
  // data rows
  c.values.forEach((row, i) => {
    grid.innerHTML += `<div class="corr-label">${c.columns[i].length > 8 ? c.columns[i].slice(0,7) + '\u2026' : c.columns[i]}</div>`;
    row.forEach(v => {
      const intensity = Math.abs(v);
      const hue = v >= 0 ? 200 : 0;
      grid.innerHTML += `<div class="corr-cell" style="background:hsla(${hue},70%,50%,${intensity * 0.7 + 0.05})">${v.toFixed(2)}</div>`;
    });
  });
} else {
  document.getElementById('corr-card').innerHTML =
    '<h2>Correlation Matrix</h2><p style="color:#64748b;padding:20px;">No numeric features to correlate.</p>';
}

// --- Target distribution ---
if (A.target_distribution.type === 'numeric') {
  const h = A.target_distribution.histogram;
  const labels = h.edges.slice(0, -1).map((e, i) => ((e + h.edges[i+1]) / 2).toFixed(1));
  new Chart(document.getElementById('dist-chart'), {
    type: 'bar',
    data: { labels, datasets: [{ data: h.counts, backgroundColor: '#38bdf8', borderRadius: 4 }]},
    options: { responsive: true, plugins: { legend: { display: false } },
      scales: { y: { title: { display: true, text: 'Count' }}}}
  });
} else {
  new Chart(document.getElementById('dist-chart'), {
    type: 'doughnut',
    data: { labels: A.target_distribution.labels,
      datasets: [{ data: A.target_distribution.counts,
        backgroundColor: A.target_distribution.labels.map((_, i) =>
          `hsla(${i * 360 / A.target_distribution.labels.length}, 70%, 60%, 0.8)`)
      }]},
    options: { responsive: true }
  });
}

// --- Feature table ---
const tbody = document.querySelector('#feature-table tbody');
A.feature_stats.forEach(f => {
  const detail = f.type === 'numeric'
    ? `mean=${f.mean}, std=${f.std}`
    : `${f.unique} unique, top="${f.top}"`;
  const badge = f.type === 'numeric' ? 'badge-blue' : 'badge-purple';
  tbody.innerHTML += `<tr>
    <td><strong>${f.name}</strong></td>
    <td><span class="badge ${badge}">${f.type}</span></td>
    <td>${detail}</td>
    <td>${f.missing}</td>
  </tr>`;
});

// --- API box ---
const baseUrl = window.location.origin;
document.getElementById('api-box').innerHTML =
  `curl -X POST ${baseUrl}/predict \\<br>
  &nbsp;&nbsp;-H "Content-Type: application/json" \\<br>
  &nbsp;&nbsp;-d '{"data": [{"feature": "value"}]}'`;
</script>
</body>
</html>"""
