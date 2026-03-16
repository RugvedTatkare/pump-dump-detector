// ── Config ────────────────────────────────────────────────────
const API_BASE = window.location.hostname === 'localhost'
  ? 'http://localhost:8000'
  : 'https://pump-dump-detector-api.onrender.com'; // ← update after deploy

// ── State ─────────────────────────────────────────────────────
let selectedCoin = 'BTC';
let selectedDays = 7;
let mainChart = null;
let compChart = null;
let activeChartTab = 'timeline';
let lastComponentScores = null;

// ── DOM ───────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

// ── Init ──────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initCharts();
  initCoinButtons();
  initRangeButtons();
  initChartTabs();

  $('analyze-btn').addEventListener('click', runAnalysis);
  $('simulate-btn').addEventListener('click', runSimulation);
  $('refresh-alerts-btn').addEventListener('click', fetchAlerts);
});

// ── Coin Buttons ──────────────────────────────────────────────
function initCoinButtons() {
  document.querySelectorAll('.coin-row').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.coin-row').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      selectedCoin = btn.dataset.coin;
    });
  });
}

// ── Range Buttons ─────────────────────────────────────────────
function initRangeButtons() {
  document.querySelectorAll('.range-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.range-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      selectedDays = parseInt(btn.dataset.days);
    });
  });
}

// ── Chart Tabs ────────────────────────────────────────────────
function initChartTabs() {
  document.querySelectorAll('.chart-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.chart-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      activeChartTab = tab.dataset.chart;

      if (activeChartTab === 'components') {
        $('main-chart').classList.add('hidden');
        $('comp-chart').classList.remove('hidden');
        if (lastComponentScores) renderComponentChart(lastComponentScores);
      } else {
        $('comp-chart').classList.add('hidden');
        $('main-chart').classList.remove('hidden');
      }
    });
  });
}

// ── Charts Setup ──────────────────────────────────────────────
function initCharts() {
  // Main timeline chart
  const ctx1 = $('main-chart').getContext('2d');
  mainChart = new Chart(ctx1, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Price',
          data: [],
          borderColor: '#00ff88',
          backgroundColor: 'rgba(0,255,136,0.04)',
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.35,
          yAxisID: 'yPrice',
          fill: true,
        },
        {
          label: 'Pump Prob',
          data: [],
          borderColor: '#ffb340',
          backgroundColor: 'rgba(255,179,64,0.04)',
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.35,
          yAxisID: 'yProb',
          borderDash: [4, 3],
          fill: true,
        },
      ],
    },
    options: chartOptions('yPrice', 'yProb'),
  });

  // Component bar chart
  const ctx2 = $('comp-chart').getContext('2d');
  compChart = new Chart(ctx2, {
    type: 'bar',
    data: {
      labels: ['Random Walk', 'HMM Regime', 'Poisson Jumps', 'Volume'],
      datasets: [{
        label: 'Score',
        data: [0, 0, 0, 0],
        backgroundColor: ['rgba(0,212,255,0.6)', 'rgba(0,255,136,0.6)', 'rgba(255,179,64,0.6)', 'rgba(255,122,48,0.6)'],
        borderColor: ['#00d4ff', '#00ff88', '#ffb340', '#ff7a30'],
        borderWidth: 1,
        borderRadius: 3,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0b0f18',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          titleColor: '#6b7e99',
          bodyColor: '#c8d4e8',
          callbacks: { label: ctx => ` ${(ctx.parsed.y * 100).toFixed(1)}%` },
        },
      },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#6b7e99', font: { family: 'IBM Plex Mono', size: 10 } } },
        y: { min: 0, max: 1, grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#6b7e99', font: { family: 'IBM Plex Mono', size: 10 }, callback: v => `${(v*100).toFixed(0)}%` } },
      },
    },
  });
}

function chartOptions(yPriceId, yProbId) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#0b0f18',
        borderColor: 'rgba(255,255,255,0.1)',
        borderWidth: 1,
        titleColor: '#6b7e99',
        bodyColor: '#c8d4e8',
        callbacks: {
          label: ctx => {
            if (ctx.datasetIndex === 0) return ` $${Number(ctx.parsed.y).toFixed(4)}`;
            return ` ${(ctx.parsed.y * 100).toFixed(1)}% pump prob`;
          },
        },
      },
    },
    scales: {
      x: {
        display: true,
        grid: { color: 'rgba(255,255,255,0.04)' },
        ticks: { color: '#6b7e99', font: { family: 'IBM Plex Mono', size: 9 }, maxTicksLimit: 8 },
      },
      yPrice: {
        position: 'left',
        grid: { color: 'rgba(255,255,255,0.04)' },
        ticks: { color: '#00ff88', font: { family: 'IBM Plex Mono', size: 9 } },
      },
      yProb: {
        position: 'right',
        min: 0, max: 1,
        grid: { display: false },
        ticks: { color: '#ffb340', font: { family: 'IBM Plex Mono', size: 9 }, callback: v => `${(v*100).toFixed(0)}%` },
      },
    },
  };
}

function renderComponentChart(scores) {
  if (!compChart) return;
  compChart.data.datasets[0].data = [
    scores.random_walk || 0,
    scores.hmm_regime  || 0,
    scores.poisson_jumps || 0,
    scores.volume_anomaly || 0,
  ];
  compChart.update();
}

// ── Analysis ──────────────────────────────────────────────────
async function runAnalysis() {
  setLoading(true);
  setStatus(`SCANNING ${selectedCoin}...`);

  try {
    const [analysisRes, histRes] = await Promise.all([
      fetch(`${API_BASE}/api/analyze/${selectedCoin}?days=${selectedDays}`),
      fetch(`${API_BASE}/api/historical/${selectedCoin}?days=${selectedDays}`),
    ]);

    if (!analysisRes.ok) throw new Error(await analysisRes.text());

    const analysis = await analysisRes.json();
    updateAllPanels(analysis);
    setStatus(`${selectedCoin} · ${new Date().toLocaleTimeString()}`);

    if (histRes.ok) {
      const hist = await histRes.json();
      renderTimeline(hist);
    }

    await fetchAlerts();
  } catch (err) {
    setStatus(`ERR: ${err.message}`);
    console.error(err);
  } finally {
    setLoading(false);
  }
}

async function runSimulation() {
  setLoading(true);
  setStatus('SIMULATING PUMP...');

  try {
    const res = await fetch(`${API_BASE}/api/simulate?scenario=pump&steps=60`);
    const data = await res.json();
    data.symbol = selectedCoin + ' (SIM)';
    updateAllPanels(data);

    const path = data.simulated_path || [];
    const labels = path.map((_, i) => `t+${i}`);
    const probs  = Array(path.length).fill(data.pump_probability);
    updateChartDirect(labels, path, probs);
    setStatus(`SIMULATION · pump scenario`);
  } catch (err) {
    setStatus(`SIM ERR: ${err.message}`);
  } finally {
    setLoading(false);
  }
}

// ── Panel Updates ─────────────────────────────────────────────
function updateAllPanels(data) {
  updateScore(data);
  updateComponents(data);
  updateHeader(data);
  updateSignals(data);
}

function updateScore(data) {
  const score = data.pump_probability || 0;
  const risk  = data.risk_level || 'LOW';
  const pm    = data.price_metrics || {};

  const pct = score * 100;
  $('score-value').textContent = `${pct.toFixed(1)}%`;

  const circle = $('ring-circle');
  const circumference = 314;
  const offset = circumference - (score * circumference);
  circle.style.strokeDashoffset = offset;

  const riskColors = { LOW: '#00ff88', MEDIUM: '#ffb340', HIGH: '#ff7a30', CRITICAL: '#ff4d4d' };
  circle.style.stroke = riskColors[risk] || '#00ff88';
  $('score-pct').style.color = riskColors[risk] || '#00ff88';

  const riskClass = { LOW: '', MEDIUM: 'risk-medium', HIGH: 'risk-high', CRITICAL: 'risk-critical' }[risk] || '';
  $('main-score-card').className = `score-card ${riskClass}`;
  $('risk-badge').className = `risk-badge ${riskClass}`;
  $('risk-badge').textContent = risk;

  $('score-symbol').textContent = `${data.symbol || selectedCoin} / USD`;

  setChange('met-1h',  pm.change_1h_pct);
  setChange('met-24h', pm.change_24h_pct);
}

function updateComponents(data) {
  const scores = data.component_scores || {};
  lastComponentScores = scores;

  setBar('rw',  scores.random_walk    || 0);
  setBar('hmm', scores.hmm_regime     || 0);
  setBar('pj',  scores.poisson_jumps  || 0);
  setBar('vol', scores.volume_anomaly || 0);

  const stateProbs = data.hmm_regime?.state_probs || [0.33, 0.33, 0.34];
  setRegime('normal', stateProbs[0] || 0);
  setRegime('trend',  stateProbs[1] || 0);
  setRegime('pump',   stateProbs[2] || 0);

  if (activeChartTab === 'components') renderComponentChart(scores);
}

function updateHeader(data) {
  const pm   = data.price_metrics || {};
  const risk = data.risk_level || 'LOW';
  const sym  = data.symbol || selectedCoin;

  $('hdr-coin').textContent  = `${sym} / USD`;
  $('hdr-price').textContent = pm.current_price ? `$${pm.current_price.toFixed(4)}` : '——';

  const riskColors = { LOW: '#00ff88', MEDIUM: '#ffb340', HIGH: '#ff7a30', CRITICAL: '#ff4d4d' };
  const hdrRisk = $('hdr-risk');
  hdrRisk.textContent = `${risk} · ${(data.pump_probability * 100).toFixed(1)}%`;
  hdrRisk.style.color = riskColors[risk] || '#00ff88';

  $('chart-title').textContent = `${sym} — PRICE & PUMP PROBABILITY TIMELINE`;
}

function updateSignals(data) {
  const pj  = data.poisson_jumps || {};
  const vol = data.volume        || {};
  const hmm = data.hmm_regime    || {};
  const rw  = data.random_walk   || {};

  setSig('sig-jump',   pj.recent_jump ? 'YES ⚡' : 'NO', pj.recent_jump ? 'danger' : '');
  setSig('sig-vol',    vol.anomalous ? `${vol.ratio?.toFixed(1)}x ⚠` : vol.ratio ? `${vol.ratio.toFixed(1)}x` : '—', vol.anomalous ? 'warning' : '');
  setSig('sig-regime', hmm.regime?.toUpperCase() || '—', hmm.regime === 'pump' ? 'danger' : hmm.regime === 'trending' ? 'warning' : '');
  setSig('sig-z',      rw.z_score !== undefined ? rw.z_score.toFixed(3) : '—', Math.abs(rw.z_score || 0) > 2.5 ? 'danger' : '');
  setSig('sig-lambda', pj.jump_intensity !== undefined ? pj.jump_intensity.toFixed(4) : '—', pj.jump_intensity > 0.2 ? 'warning' : '');
  setSig('sig-pts',    data.data_points_used || '—', '');
}

function setBar(key, value) {
  const pct = Math.round(value * 100);
  $(`bar-${key}`).style.width = `${pct}%`;
  $(`val-${key}`).textContent = `${pct}%`;
}

function setRegime(key, value) {
  const pct = Math.round(value * 100);
  $(`regime-${key}`).style.width = `${pct}%`;
  $(`regime-${key}-pct`).textContent = `${pct}%`;
}

function setChange(id, value) {
  const el = $(id);
  if (value == null) { el.textContent = '—'; el.className = 'change-val'; return; }
  el.textContent = `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  el.className   = `change-val ${value >= 0 ? 'positive' : 'negative'}`;
}

function setSig(id, text, cls) {
  const el = $(id);
  el.textContent = text;
  el.className   = `sig-val ${cls}`;
}

// ── Chart Rendering ───────────────────────────────────────────
function renderTimeline(hist) {
  if (!hist?.timeline) return;
  const tl     = hist.timeline;
  const labels = tl.map((_, i) => `W${i + 1}`);
  const probs  = tl.map(t => t.score);
  updateChartDirect(labels, probs.map(() => null), probs);
}

function updateChartDirect(labels, prices, probs) {
  mainChart.data.labels           = labels;
  mainChart.data.datasets[0].data = prices;
  mainChart.data.datasets[1].data = probs;
  mainChart.update('active');
}

// ── Alerts ────────────────────────────────────────────────────
async function fetchAlerts() {
  try {
    const res  = await fetch(`${API_BASE}/api/alerts?limit=20`);
    const data = await res.json();
    renderAlerts(data.alerts || [], data.stats || {});
  } catch { /* silent */ }
}

function renderAlerts(alerts, stats) {
  const by = stats.by_risk_level || {};
  $('stat-critical').textContent = `CRIT ${by.CRITICAL || 0}`;
  $('stat-high').textContent     = `HIGH ${by.HIGH || 0}`;
  $('stat-medium').textContent   = `MED ${by.MEDIUM || 0}`;

  const list = $('alerts-list');
  if (!alerts.length) {
    list.innerHTML = '<div class="empty-state">No alerts generated — risk must reach MEDIUM+ to trigger.</div>';
    return;
  }

  list.innerHTML = alerts.map(a => {
    const ts = new Date(a.timestamp).toLocaleTimeString();
    return `
      <div class="alert-item">
        <span class="alert-level ${a.risk_level}">${a.risk_level}</span>
        <div class="alert-body">
          <div class="alert-title">${a.symbol} · ${(a.pump_probability * 100).toFixed(1)}% pump prob</div>
          <div class="alert-msg">${a.message}</div>
        </div>
        <div class="alert-time">${ts}</div>
      </div>`;
  }).join('');
}

// ── Helpers ───────────────────────────────────────────────────
function setLoading(on) {
  $('analyze-btn').disabled = on;
  $('btn-text').textContent = on ? 'SCANNING...' : '▶ RUN ANALYSIS';
  $('btn-spinner').classList.toggle('hidden', !on);
}

function setStatus(msg) { $('status-text').textContent = msg; }