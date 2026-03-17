import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
import asyncio

st.set_page_config(
    page_title="PumpSentinel — Crypto Manipulation Detector",
    page_icon="🔺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
:root{--bg:#070a0f;--bg1:#0b0f18;--bg2:#0f1520;--bg3:#141c2a;--bg4:#1a2335;--border:rgba(255,255,255,0.06);--border2:rgba(255,255,255,0.11);--text:#c8d4e8;--text2:#6b7e99;--text3:#3d4e63;--green:#00ff88;--amber:#ffb340;--orange:#ff7a30;--red:#ff4d4d;--blue:#38b2ff;--mono:'IBM Plex Mono',monospace;--sans:'IBM Plex Sans',sans-serif}
html,body,[data-testid="stAppViewContainer"],[data-testid="stApp"]{background:var(--bg)!important;color:var(--text)!important;font-family:var(--sans)!important}
[data-testid="stSidebar"]{background:var(--bg1)!important;border-right:1px solid var(--border2)!important}
[data-testid="stSidebar"] *{color:var(--text)!important}
[data-testid="stHeader"]{display:none!important}
[data-testid="stToolbar"]{display:none!important}
[data-testid="stDecoration"]{display:none!important}
.block-container{padding:0 1.5rem 1rem 1.5rem!important;background:var(--bg)!important}
h1,h2,h3,h4{font-family:var(--mono)!important;color:var(--text)!important}
hr{border-color:var(--border2)!important}
.stButton>button{background:transparent!important;border:1px solid var(--green)!important;color:var(--green)!important;font-family:var(--mono)!important;font-size:0.75rem!important;letter-spacing:0.1em!important;border-radius:3px!important;width:100%!important;padding:0.6rem!important}
.stButton>button:hover{background:rgba(0,255,136,0.08)!important}
.metric-card{background:var(--bg2);border:1px solid var(--border2);border-radius:4px;padding:1rem;font-family:var(--mono)}
.metric-label{font-size:0.6rem;letter-spacing:0.15em;color:var(--text3);margin-bottom:0.4rem}
.metric-value{font-size:1.6rem;font-weight:600;color:var(--green)}
.panel-title{font-family:var(--mono);font-size:0.62rem;letter-spacing:0.15em;color:var(--text3);margin-bottom:1rem}
.model-row{display:flex;align-items:center;margin-bottom:0.75rem}
.model-name{font-family:var(--mono);font-size:0.75rem;font-weight:600;color:var(--text);width:140px;flex-shrink:0}
.model-sub{font-family:var(--sans);font-size:0.65rem;color:var(--text3)}
.model-bar-wrap{flex:1;height:3px;background:var(--bg4);margin:0 1rem;border-radius:2px;overflow:hidden}
.model-bar-fill{height:100%;border-radius:2px}
.model-pct{font-family:var(--mono);font-size:0.7rem;color:var(--text2);width:36px;text-align:right}
.signal-row{display:flex;justify-content:space-between;align-items:center;padding:0.35rem 0;border-bottom:1px solid var(--border);font-family:var(--mono);font-size:0.72rem}
.signal-row:last-child{border-bottom:none}
.sig-label{color:var(--text2)}
.sig-val{color:var(--text)}
.sig-val.green{color:var(--green)}
.sig-val.amber{color:var(--amber)}
.sig-val.red{color:var(--red)}
.header-bar{display:flex;align-items:center;justify-content:space-between;background:var(--bg1);border-bottom:1px solid var(--border2);padding:0.6rem 1.5rem;margin:0 -1.5rem 1rem -1.5rem;font-family:var(--mono);font-size:0.78rem}
.logo{display:flex;align-items:center;gap:0.5rem;font-weight:700;letter-spacing:0.1em;color:var(--text)}
.logo-accent{color:var(--green)}
.live-dot{width:6px;height:6px;border-radius:50%;background:var(--green);display:inline-block;margin-right:4px;animation:blink 2s ease-in-out infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0.3}}
</style>
""", unsafe_allow_html=True)

# ── Models ────────────────────────────────────────────────────────────────────
class RandomWalkModel:
    def __init__(self, window=20): self.window = window
    def compute_log_returns(self, prices): return np.diff(np.log(np.array(prices, dtype=float)))
    def estimate_params(self, lr):
        w = lr[-self.window:] if len(lr) >= self.window else lr
        return float(np.mean(w)), float(np.std(w)) + 1e-9
    def abnormality_score(self, prices):
        if len(prices) < 3: return {"z_score": 0.0, "mu": 0.0, "sigma": 0.0, "is_abnormal": False, "latest_log_return": 0.0}
        lr = self.compute_log_returns(prices)
        mu, sigma = self.estimate_params(lr[:-1])
        z = (lr[-1] - mu) / sigma
        return {"z_score": round(z, 4), "mu": round(mu, 6), "sigma": round(sigma, 6),
                "latest_log_return": round(lr[-1], 6), "is_abnormal": abs(z) > 2.5}

class PoissonJumpModel:
    def __init__(self, jump_threshold_sigma=2.5): self.jump_threshold_sigma = jump_threshold_sigma
    def detect_jumps(self, log_returns):
        if len(log_returns) < 5: return {"jump_intensity": 0.0, "positive_jump_ratio": 0.5, "n_jumps": 0, "jump_sizes": [], "recent_jump": False, "recent_z_score": 0.0}
        mu = np.mean(log_returns); sigma = np.std(log_returns) + 1e-9
        z_scores = (log_returns - mu) / sigma
        jump_mask = np.abs(z_scores) > self.jump_threshold_sigma
        jump_sizes = log_returns[jump_mask].tolist()
        n_jumps = int(np.sum(jump_mask))
        lam = n_jumps / len(log_returns)
        pos_ratio = sum(1 for j in jump_sizes if j > 0) / n_jumps if n_jumps > 0 else 0.5
        return {"jump_intensity": round(lam, 4), "positive_jump_ratio": round(pos_ratio, 4),
                "n_jumps": n_jumps, "jump_sizes": [round(j, 6) for j in jump_sizes[-10:]],
                "recent_jump": bool(jump_mask[-1]), "recent_z_score": round(float(z_scores[-1]), 3)}
    def pump_score_from_jumps(self, lam, pos_ratio, recent_jump):
        i_score = 1 - np.exp(-5 * lam)
        d_score = max(0.0, (pos_ratio - 0.5) * 2)
        return round(min(1.0, max(0.0, 0.5 * i_score + 0.35 * d_score + (0.15 if recent_jump else 0))), 4)
    def simulate_jump_path(self, S0, mu, sigma, lam, mu_j, sigma_j, steps=60):
        prices = [S0]
        for _ in range(steps):
            d = (mu - 0.5 * sigma**2) + sigma * np.random.normal()
            j = sum(np.random.normal(mu_j, sigma_j) for _ in range(np.random.poisson(lam)))
            prices.append(float(prices[-1] * np.exp(d + j)))
        return prices

class HMMRegimeModel:
    def __init__(self, n_states=3): self.n_states = n_states; self.is_fitted = False
    def fit(self, log_returns):
        try:
            from hmmlearn import hmm
            if len(log_returns) < 20: return False
            X = log_returns.reshape(-1, 1)
            self.model = hmm.GaussianHMM(n_components=self.n_states, covariance_type="full", n_iter=200, random_state=42, tol=1e-4)
            self.model.fit(X); self.is_fitted = True
            stds = [np.sqrt(self.model.covars_[i][0][0]) for i in range(self.n_states)]
            self._remap = {old: new for new, old in enumerate(np.argsort(stds))}
            return True
        except: self.is_fitted = False; return False
    def predict_state(self, log_returns):
        if not self.is_fitted or len(log_returns) < 5:
            return {"current_state": 0, "state_probs": [0.8, 0.15, 0.05], "regime": "normal", "pump_probability": 0.05}
        try:
            X = log_returns.reshape(-1, 1)
            _, posteriors = self.model.score_samples(X)
            cp = posteriors[-1].tolist()
            _, states = self.model.decode(X, algorithm="viterbi")
            cs = self._remap.get(int(states[-1]), int(states[-1]))
            rp = [0.0] * self.n_states
            for i, p in enumerate(cp): rp[self._remap.get(i, i)] += p
            return {"current_state": cs, "state_probs": [round(p, 4) for p in rp],
                    "regime": {0:"normal",1:"trending",2:"pump"}.get(cs,"unknown"), "pump_probability": round(float(rp[2]), 4)}
        except: return {"current_state": 0, "state_probs": [0.8, 0.15, 0.05], "regime": "normal", "pump_probability": 0.05}

def analyze(prices, volumes=None):
    if len(prices) < 5: return None
    rw = RandomWalkModel(window=20); hmm = HMMRegimeModel(n_states=3); pj = PoissonJumpModel(jump_threshold_sigma=2.5)
    lr = rw.compute_log_returns(prices)
    rw_result = rw.abnormality_score(prices)
    rw_score = float(1 - np.exp(-abs(rw_result["z_score"]) / 3.0))
    if len(lr) >= 20: hmm.fit(lr)
    hmm_result = hmm.predict_state(lr); hmm_score = hmm_result["pump_probability"]
    pj_result = pj.detect_jumps(lr)
    pj_score = pj.pump_score_from_jumps(pj_result["jump_intensity"], pj_result["positive_jump_ratio"], pj_result["recent_jump"])
    vol_score, vol_ratio = 0.0, 1.0
    if volumes and len(volumes) >= 5:
        bv = float(np.mean(volumes[:-1])); cv = float(volumes[-1])
        if bv > 0: vol_ratio = cv / bv; vol_score = min(1.0, (vol_ratio - 1.0) / 5.0)
    base = 0.30 * rw_score + 0.40 * hmm_score + 0.30 * pj_score
    final = min(1.0, base * (1.0 + 0.3 * vol_score))
    risk = "LOW" if final < 0.35 else "MEDIUM" if final < 0.65 else "HIGH" if final < 0.80 else "CRITICAL"
    p1h = (prices[-1] / prices[-2] - 1) * 100 if len(prices) >= 2 else 0
    p24h = (prices[-1] / prices[0] - 1) * 100 if len(prices) >= 2 else 0
    return {"pump_probability": round(float(final), 4), "risk_level": risk,
            "component_scores": {"random_walk": round(rw_score, 4), "hmm_regime": round(hmm_score, 4), "poisson_jumps": round(pj_score, 4), "volume_anomaly": round(vol_score, 4)},
            "random_walk": rw_result, "hmm_regime": hmm_result, "poisson_jumps": pj_result,
            "volume": {"ratio": round(vol_ratio, 3), "anomalous": vol_ratio > 3.0},
            "price_metrics": {"current_price": round(prices[-1], 6), "change_1h_pct": round(p1h, 3), "change_24h_pct": round(p24h, 3)}}

async def fetch_coingecko(symbol, days=7):
    SYMBOL_MAP = {"BTC":"bitcoin","ETH":"ethereum","SOL":"solana","DOGE":"dogecoin","SHIB":"shiba-inu","ADA":"cardano","XRP":"ripple","MATIC":"matic-network","PEPE":"pepe","BNB":"binancecoin"}
    url = f"https://api.coingecko.com/api/v3/coins/{SYMBOL_MAP.get(symbol.upper(), symbol.lower())}/market_chart"
    api_key = st.secrets.get("COINGECKO_API_KEY", "")
    st.write(f"Key found: {bool(api_key)}")  # debug - remove after
    headers = {"x-cg-demo-api-key": api_key} if api_key else {}
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url, params={"vs_currency": "usd", "days": days}, headers=headers)
        resp.raise_for_status(); data = resp.json()
        return [p[1] for p in data["prices"]], [v[1] for v in data["total_volumes"]]

def risk_hex(risk): return {"LOW":"#00ff88","MEDIUM":"#ffb340","HIGH":"#ff7a30","CRITICAL":"#ff4d4d"}.get(risk,"#c8d4e8")
def bar_color(v):
    if v > 0.75: return "#ff4d4d"
    if v > 0.50: return "#ff7a30"
    if v > 0.25: return "#ffb340"
    return "#00ff88"

def make_gauge(prob, risk):
    color = risk_hex(risk)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"family": "IBM Plex Mono", "size": 28, "color": color}},
        gauge={"axis": {"range": [0, 100], "tickfont": {"size": 9, "color": "#3d4e63"}, "tickcolor": "#3d4e63"},
               "bar": {"color": color, "thickness": 0.25}, "bgcolor": "#0f1520",
               "borderwidth": 1, "bordercolor": "#1a2335",
               "steps": [{"range": [0,35],"color":"rgba(0,255,136,0.05)"},{"range":[35,65],"color":"rgba(255,179,64,0.05)"},
                         {"range":[65,80],"color":"rgba(255,122,48,0.05)"},{"range":[80,100],"color":"rgba(255,77,77,0.05)"}],
               "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.75, "value": prob * 100}}))
    fig.update_layout(height=200, margin=dict(l=20,r=20,t=20,b=10), paper_bgcolor="#0b0f18",
                      font={"family":"IBM Plex Mono"},
                      annotations=[{"text":"PUMP PROB","x":0.5,"y":0.1,"showarrow":False,
                                     "font":{"size":9,"color":"#3d4e63","family":"IBM Plex Mono"},"xanchor":"center"}])
    return fig

def make_price_chart(prices, pump_probs=None, risk="LOW"):
    color = risk_hex(risk)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.02)
    fig.add_trace(go.Scatter(y=prices, mode="lines", line=dict(color=color, width=1.5),
                              fill="tozeroy", fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.05)",
                              name="Price", showlegend=True), row=1, col=1)
    if pump_probs:
        fig.add_trace(go.Scatter(y=[p*100 for p in pump_probs], mode="lines",
                                  line=dict(color="#38b2ff", width=1.2, dash="dot"),
                                  name="Pump Prob %", showlegend=True), row=2, col=1)
    fig.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor="#0b0f18", plot_bgcolor="#0b0f18",
                      font={"family":"IBM Plex Mono","size":10,"color":"#6b7e99"},
                      legend=dict(orientation="h",x=1,xanchor="right",y=1.05,font=dict(size=9,color="#6b7e99"),bgcolor="rgba(0,0,0,0)"),
                      yaxis=dict(showgrid=True,gridcolor="rgba(255,255,255,0.04)",zeroline=False,tickfont=dict(size=9),side="right"),
                      yaxis2=dict(showgrid=True,gridcolor="rgba(255,255,255,0.03)",zeroline=False,tickfont=dict(size=9),range=[0,100],side="right"))
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    return fig

# ── Session State ─────────────────────────────────────────────────────────────
for k, v in [("result",None),("prices",[]),("pump_probs",[]),("selected_coin","BTC"),("selected_days",7)]:
    if k not in st.session_state: st.session_state[k] = v

result = st.session_state.result
prices = st.session_state.prices

# ── Header ────────────────────────────────────────────────────────────────────
coin_d = st.session_state.selected_coin
price_d = f"${result['price_metrics']['current_price']:,.6f}" if result else "——"
risk_d = result["risk_level"] if result else "AWAITING"
rc = risk_hex(risk_d) if result else "#6b7e99"

st.markdown(f"""
<div class="header-bar">
  <div class="logo">
    <svg width="18" height="18" viewBox="0 0 22 22" fill="none">
      <polygon points="11,1 21,6.5 21,15.5 11,21 1,15.5 1,6.5" stroke="#00ff88" stroke-width="1.4" fill="none"/>
      <polygon points="11,5 17,8.5 17,13.5 11,17 5,13.5 5,8.5" stroke="#00ff88" stroke-width="0.8" fill="rgba(0,255,136,0.06)"/>
    </svg>
    PUMP<span class="logo-accent">SENTINEL</span>
  </div>
  <div style="display:flex;align-items:center;gap:1.5rem;font-size:0.75rem;color:#6b7e99">
    <span style="color:#c8d4e8">{coin_d} / USD</span><span style="color:#3d4e63">|</span>
    <span style="color:#c8d4e8">{price_d}</span><span style="color:#3d4e63">|</span>
    <span style="color:{rc};font-weight:600">{risk_d}</span>
  </div>
  <div style="display:flex;align-items:center;gap:6px;font-size:0.7rem;color:#6b7e99">
    <span class="live-dot"></span> LIVE
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
COINS = [("BTC","Bitcoin"),("ETH","Ethereum"),("SOL","Solana"),("DOGE","Dogecoin"),
         ("SHIB","Shiba Inu"),("PEPE","Pepe"),("ADA","Cardano"),("XRP","Ripple")]

with st.sidebar:
    st.markdown('<div class="panel-title">MARKET</div>', unsafe_allow_html=True)
    selected = st.selectbox("", [s for s,_ in COINS],
                             index=[s for s,_ in COINS].index(st.session_state.selected_coin),
                             label_visibility="collapsed")
    st.session_state.selected_coin = selected

    coin_html = ""
    for sym, name in COINS:
        color = "#00ff88" if sym == selected else "#3d4e63"
        bg = "rgba(0,255,136,0.07)" if sym == selected else "transparent"
        coin_html += f'<div style="display:flex;justify-content:space-between;padding:0.4rem 0.5rem;border-radius:3px;background:{bg};font-family:IBM Plex Mono,monospace;font-size:0.72rem;margin-bottom:2px"><span style="color:{color}">{sym}</span><span style="color:#3d4e63;font-size:0.65rem">{name}</span></div>'
    st.markdown(coin_html, unsafe_allow_html=True)

    st.markdown('<hr style="border-color:rgba(255,255,255,0.06);margin:0.6rem 0">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">TIMEFRAME</div>', unsafe_allow_html=True)
    days_map = {"1D":1,"7D":7,"30D":30}
    days_sel = st.radio("", list(days_map.keys()), index=1, horizontal=True, label_visibility="collapsed")
    st.session_state.selected_days = days_map[days_sel]

    st.markdown('<hr style="border-color:rgba(255,255,255,0.06);margin:0.6rem 0">', unsafe_allow_html=True)
    run_live = st.button("▶  RUN ANALYSIS")
    run_sim_pump = st.button("⚡  SIMULATE PUMP")
    run_sim_normal = st.button("〜  SIMULATE NORMAL")

    st.markdown('<hr style="border-color:rgba(255,255,255,0.06);margin:0.6rem 0">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">SIGNALS</div>', unsafe_allow_html=True)
    if result:
        pj=result["poisson_jumps"]; hmm_r=result["hmm_regime"]; rw_r=result["random_walk"]; vol=result["volume"]
        jc="red" if pj["recent_jump"] else "green"
        vc="red" if vol["anomalous"] else "green"
        rc2="red" if hmm_r["regime"]=="pump" else "amber" if hmm_r["regime"]=="trending" else "green"
        zc="red" if abs(rw_r["z_score"])>2.5 else "green"
        st.markdown(f"""
        <div class="signal-row"><span class="sig-label">Jump Event</span><span class="sig-val {jc}">{"YES" if pj["recent_jump"] else "NO"}</span></div>
        <div class="signal-row"><span class="sig-label">Vol. Spike</span><span class="sig-val {vc}">{vol["ratio"]:.2f}x</span></div>
        <div class="signal-row"><span class="sig-label">Regime</span><span class="sig-val {rc2}">{hmm_r["regime"].upper()}</span></div>
        <div class="signal-row"><span class="sig-label">Z-Score</span><span class="sig-val {zc}">{rw_r["z_score"]:.3f}</span></div>
        <div class="signal-row"><span class="sig-label">Jump λ</span><span class="sig-val">{pj["jump_intensity"]:.4f}</span></div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="signal-row"><span class="sig-label">Jump Event</span><span class="sig-val">—</span></div>
        <div class="signal-row"><span class="sig-label">Vol. Spike</span><span class="sig-val">—</span></div>
        <div class="signal-row"><span class="sig-label">Regime</span><span class="sig-val">—</span></div>
        <div class="signal-row"><span class="sig-label">Z-Score</span><span class="sig-val">—</span></div>
        <div class="signal-row"><span class="sig-label">Jump λ</span><span class="sig-val">—</span></div>
        """, unsafe_allow_html=True)

# ── Run Logic ─────────────────────────────────────────────────────────────────
def compute_rolling_probs(prices_new):
    probs = []
    for i in range(10, len(prices_new)):
        r = analyze(prices_new[max(0,i-24):i+1])
        probs.append(r["pump_probability"] if r else 0)
    return probs

if run_live:
    with st.spinner("Fetching live data..."):
        try:
            prices_new, volumes = asyncio.run(fetch_coingecko(st.session_state.selected_coin, st.session_state.selected_days))
            result_new = analyze(prices_new, volumes)
            if result_new:
                st.session_state.prices = prices_new
                st.session_state.result = result_new
                st.session_state.pump_probs = compute_rolling_probs(prices_new)
                st.rerun()
        except Exception as e:
            st.error(f"CoinGecko error: {e}. Try Simulate instead.")
elif run_sim_pump:
    prices_new = PoissonJumpModel().simulate_jump_path(S0=100.0, mu=0.08, sigma=0.12, lam=0.3, mu_j=0.15, sigma_j=0.05, steps=80)
    result_new = analyze(prices_new)
    if result_new:
        st.session_state.prices = prices_new
        st.session_state.result = result_new
        st.session_state.pump_probs = compute_rolling_probs(prices_new)
        st.rerun()
elif run_sim_normal:
    prices_new = PoissonJumpModel().simulate_jump_path(S0=100.0, mu=0.001, sigma=0.02, lam=0.02, mu_j=0.0, sigma_j=0.01, steps=80)
    result_new = analyze(prices_new)
    if result_new:
        st.session_state.prices = prices_new
        st.session_state.result = result_new
        st.session_state.pump_probs = compute_rolling_probs(prices_new)
        st.rerun()

# ── Main Content ──────────────────────────────────────────────────────────────
result = st.session_state.result
prices = st.session_state.prices
pump_probs = st.session_state.pump_probs

if result is None:
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:60vh;font-family:'IBM Plex Mono',monospace;color:#3d4e63;text-align:center">
        <div style="font-size:2rem;margin-bottom:1rem">◈</div>
        <div style="font-size:0.8rem;letter-spacing:0.2em">AWAITING DATA</div>
        <div style="font-size:0.65rem;margin-top:0.5rem;color:#2a3545">Select a token and click RUN ANALYSIS</div>
    </div>""", unsafe_allow_html=True)
else:
    prob=result["pump_probability"]; risk=result["risk_level"]; scores=result["component_scores"]
    hmm_r=result["hmm_regime"]; rw_r=result["random_walk"]; pj_r=result["poisson_jumps"]

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.plotly_chart(make_gauge(prob, risk), use_container_width=True, config={"displayModeBar":False})
        rc3=risk_hex(risk); p1h=result["price_metrics"]["change_1h_pct"]
        p24h=result["price_metrics"]["change_24h_pct"]; cur=result["price_metrics"]["current_price"]
        st.markdown(f"""
        <div style="text-align:center;font-family:'IBM Plex Mono',monospace;margin-bottom:0.8rem">
            <div style="font-size:1.1rem;font-weight:700;color:{rc3};letter-spacing:0.1em">{risk}</div>
            <div style="font-size:0.72rem;color:#6b7e99;margin-top:2px">${cur:,.6f} / USD</div>
            <div style="display:flex;justify-content:center;gap:1.5rem;margin-top:0.5rem;font-size:0.7rem">
                <div><span style="color:#3d4e63">1H</span><br><span style="color:{'#00ff88' if p1h>=0 else '#ff4d4d'}">{'+' if p1h>=0 else ''}{p1h:.2f}%</span></div>
                <div><span style="color:#3d4e63">24H</span><br><span style="color:{'#00ff88' if p24h>=0 else '#ff4d4d'}">{'+' if p24h>=0 else ''}{p24h:.2f}%</span></div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="panel-title">MODEL BREAKDOWN</div>', unsafe_allow_html=True)
        for name, sub, val in [("Random Walk","GBM z-score deviation",scores["random_walk"]),
                                ("HMM Regime","Pump state posterior",scores["hmm_regime"]),
                                ("Poisson Jumps","Jump intensity + direction",scores["poisson_jumps"]),
                                ("Volume Anomaly","Current vs baseline ratio",scores["volume_anomaly"])]:
            bc=bar_color(val); pct=round(val*100,1)
            st.markdown(f'<div class="model-row"><div style="width:130px;flex-shrink:0"><div class="model-name">{name}</div><div class="model-sub">{sub}</div></div><div class="model-bar-wrap"><div class="model-bar-fill" style="width:{pct}%;background:{bc}"></div></div><div class="model-pct">{pct}%</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="panel-title" style="margin-top:1rem">REGIME STATES</div>', unsafe_allow_html=True)
        sp=hmm_r["state_probs"]
        for i,(label,color) in enumerate(zip(["Normal","Trending","Pump"],["#00ff88","#ffb340","#ff4d4d"])):
            pct=round(sp[i]*100,1) if i<len(sp) else 0
            st.markdown(f'<div class="model-row"><div style="width:80px;flex-shrink:0;font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#6b7e99">{label}</div><div class="model-bar-wrap"><div class="model-bar-fill" style="width:{pct}%;background:{color}"></div></div><div class="model-pct">{pct}%</div></div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div style="display:flex;justify-content:space-between;font-family:IBM Plex Mono,monospace;font-size:0.62rem;color:#3d4e63;letter-spacing:0.12em;margin-bottom:0.5rem"><span>— PRICE & PUMP PROBABILITY TIMELINE</span><div style="display:flex;gap:1rem"><span style="color:#6b7e99">— Price</span><span style="color:#38b2ff">--- Pump Prob</span></div></div>', unsafe_allow_html=True)
        st.plotly_chart(make_price_chart(prices, pump_probs if pump_probs else None, risk),
                        use_container_width=True, config={"displayModeBar":False})

        if risk != "LOW":
            ac=risk_hex(risk)
            abg=f"rgba({int(ac[1:3],16)},{int(ac[3:5],16)},{int(ac[5:7],16)},0.06)"
            msg={"MEDIUM":f"Elevated pump signal on {coin_d}. Score {prob*100:.1f}%. Monitor closely.",
                 "HIGH":f"High manipulation probability on {coin_d}. Score {prob*100:.1f}%. Caution advised.",
                 "CRITICAL":f"PUMP-AND-DUMP IN PROGRESS on {coin_d}. Score {prob*100:.1f}%. Do NOT buy."}.get(risk,"")
            st.markdown(f'<div style="background:{abg};border:1px solid {ac};border-radius:3px;padding:0.75rem 1rem;font-family:IBM Plex Mono,monospace;font-size:0.72rem;margin-bottom:1rem"><span style="color:{ac};font-weight:600">[{risk}]</span><span style="color:#c8d4e8;margin-left:0.5rem">{msg}</span></div>', unsafe_allow_html=True)

        rc4=risk_hex(risk)
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;font-family:IBM Plex Mono,monospace">
            <div class="metric-card"><div class="metric-label">PUMP PROB</div><div class="metric-value" style="color:{rc4}">{prob*100:.1f}%</div></div>
            <div class="metric-card"><div class="metric-label">Z-SCORE</div><div class="metric-value" style="font-size:1.3rem;color:{'#ff4d4d' if abs(rw_r['z_score'])>2.5 else '#00ff88'}">{rw_r['z_score']:.3f}</div></div>
            <div class="metric-card"><div class="metric-label">JUMP λ</div><div class="metric-value" style="font-size:1.3rem;color:{'#ff4d4d' if pj_r['jump_intensity']>0.1 else '#00ff88'}">{pj_r['jump_intensity']:.4f}</div></div>
            <div class="metric-card"><div class="metric-label">REGIME</div><div class="metric-value" style="font-size:1rem;color:{risk_hex('CRITICAL' if hmm_r['regime']=='pump' else 'MEDIUM' if hmm_r['regime']=='trending' else 'LOW')}">{hmm_r['regime'].upper()}</div></div>
        </div>""", unsafe_allow_html=True)