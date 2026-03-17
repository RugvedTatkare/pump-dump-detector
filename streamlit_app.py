import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import httpx
import asyncio
from datetime import datetime

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pump-and-Dump Detector",
    page_icon="🚨",
    layout="wide",
)

# ── Inline Models (no scipy) ──────────────────────────────────────────────────

class RandomWalkModel:
    def __init__(self, window=20):
        self.window = window

    def compute_log_returns(self, prices):
        prices = np.array(prices, dtype=float)
        return np.diff(np.log(prices))

    def estimate_params(self, log_returns):
        window_data = log_returns[-self.window:] if len(log_returns) >= self.window else log_returns
        mu = float(np.mean(window_data))
        sigma = float(np.std(window_data)) + 1e-9
        return mu, sigma

    def abnormality_score(self, prices):
        if len(prices) < 3:
            return {"z_score": 0.0, "mu": 0.0, "sigma": 0.0, "is_abnormal": False}
        log_returns = self.compute_log_returns(prices)
        mu, sigma = self.estimate_params(log_returns[:-1])
        latest_return = log_returns[-1]
        z = (latest_return - mu) / sigma
        return {
            "z_score": round(z, 4),
            "mu": round(mu, 6),
            "sigma": round(sigma, 6),
            "latest_log_return": round(latest_return, 6),
            "is_abnormal": abs(z) > 2.5,
        }


class PoissonJumpModel:
    def __init__(self, jump_threshold_sigma=2.5):
        self.jump_threshold_sigma = jump_threshold_sigma

    def detect_jumps(self, log_returns):
        if len(log_returns) < 5:
            return {"jump_intensity": 0.0, "positive_jump_ratio": 0.5, "n_jumps": 0, "jump_sizes": [], "recent_jump": False}
        mu = np.mean(log_returns)
        sigma = np.std(log_returns) + 1e-9
        z_scores = (log_returns - mu) / sigma
        jump_mask = np.abs(z_scores) > self.jump_threshold_sigma
        jump_sizes = log_returns[jump_mask].tolist()
        n_jumps = int(np.sum(jump_mask))
        T = len(log_returns)
        lam = n_jumps / T if T > 0 else 0.0
        positive_jumps = sum(1 for j in jump_sizes if j > 0)
        pos_ratio = positive_jumps / n_jumps if n_jumps > 0 else 0.5
        recent_jump = bool(jump_mask[-1]) if len(jump_mask) > 0 else False
        return {
            "jump_intensity": round(lam, 4),
            "positive_jump_ratio": round(pos_ratio, 4),
            "n_jumps": n_jumps,
            "jump_sizes": [round(j, 6) for j in jump_sizes[-10:]],
            "recent_jump": recent_jump,
            "recent_z_score": round(float(z_scores[-1]), 3),
        }

    def pump_score_from_jumps(self, jump_intensity, positive_jump_ratio, recent_jump):
        intensity_score = 1 - np.exp(-5 * jump_intensity)
        direction_score = max(0.0, (positive_jump_ratio - 0.5) * 2)
        recency_bonus = 0.15 if recent_jump else 0.0
        score = 0.5 * intensity_score + 0.35 * direction_score + recency_bonus
        return round(min(1.0, max(0.0, score)), 4)

    def simulate_jump_path(self, S0, mu, sigma, lam, mu_j, sigma_j, steps=60):
        prices = [S0]
        for _ in range(steps):
            diffusion = (mu - 0.5 * sigma**2) + sigma * np.random.normal()
            n_events = np.random.poisson(lam)
            jump = sum(np.random.normal(mu_j, sigma_j) for _ in range(n_events))
            prices.append(float(prices[-1] * np.exp(diffusion + jump)))
        return prices


class HMMRegimeModel:
    def __init__(self, n_states=3):
        self.n_states = n_states
        self.is_fitted = False

    def fit(self, log_returns):
        try:
            from hmmlearn import hmm
            if len(log_returns) < 20:
                return False
            X = log_returns.reshape(-1, 1)
            self.model = hmm.GaussianHMM(n_components=self.n_states, covariance_type="full", n_iter=200, random_state=42, tol=1e-4)
            self.model.fit(X)
            self.is_fitted = True
            stds = [np.sqrt(self.model.covars_[i][0][0]) for i in range(self.n_states)]
            sort_order = np.argsort(stds)
            self._remap = {old: new for new, old in enumerate(sort_order)}
            return True
        except Exception:
            self.is_fitted = False
            return False

    def predict_state(self, log_returns):
        if not self.is_fitted or len(log_returns) < 5:
            return {"current_state": 0, "state_probs": [1/3, 1/3, 1/3], "regime": "unknown", "pump_probability": 0.0}
        try:
            X = log_returns.reshape(-1, 1)
            log_prob, posteriors = self.model.score_samples(X)
            current_posteriors = posteriors[-1].tolist()
            _, states = self.model.decode(X, algorithm="viterbi")
            current_state_raw = int(states[-1])
            current_state = self._remap.get(current_state_raw, current_state_raw)
            remapped_probs = [0.0] * self.n_states
            for i, p in enumerate(current_posteriors):
                remapped_probs[self._remap.get(i, i)] += p
            pump_prob = remapped_probs[2]
            labels = {0: "normal", 1: "trending", 2: "pump"}
            return {
                "current_state": current_state,
                "state_probs": [round(p, 4) for p in remapped_probs],
                "regime": labels.get(current_state, "unknown"),
                "pump_probability": round(float(pump_prob), 4),
            }
        except Exception:
            return {"current_state": 0, "state_probs": [1/3, 1/3, 1/3], "regime": "normal", "pump_probability": 0.0}


WEIGHTS = {"random_walk": 0.30, "hmm": 0.40, "poisson": 0.30}

def analyze(prices, volumes=None):
    if len(prices) < 5:
        return None

    rw_model = RandomWalkModel(window=20)
    hmm_model = HMMRegimeModel(n_states=3)
    pj_model = PoissonJumpModel(jump_threshold_sigma=2.5)

    log_returns = rw_model.compute_log_returns(prices)

    rw_result = rw_model.abnormality_score(prices)
    rw_score = float(1 - np.exp(-abs(rw_result["z_score"]) / 3.0))

    if len(log_returns) >= 20:
        hmm_model.fit(log_returns)
    hmm_result = hmm_model.predict_state(log_returns)
    hmm_score = hmm_result["pump_probability"]

    pj_result = pj_model.detect_jumps(log_returns)
    pj_score = pj_model.pump_score_from_jumps(
        pj_result["jump_intensity"], pj_result["positive_jump_ratio"], pj_result["recent_jump"]
    )

    volume_score = 0.0
    volume_ratio = 1.0
    if volumes and len(volumes) >= 5:
        baseline_vol = float(np.mean(volumes[:-1]))
        current_vol = float(volumes[-1])
        if baseline_vol > 0:
            volume_ratio = current_vol / baseline_vol
            volume_score = min(1.0, (volume_ratio - 1.0) / 5.0)

    base_score = WEIGHTS["random_walk"] * rw_score + WEIGHTS["hmm"] * hmm_score + WEIGHTS["poisson"] * pj_score
    volume_multiplier = 1.0 + 0.3 * volume_score
    final_score = min(1.0, base_score * volume_multiplier)

    if final_score < 0.35:
        risk_level = "LOW"
    elif final_score < 0.65:
        risk_level = "MEDIUM"
    elif final_score < 0.80:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"

    price_change_1h = (prices[-1] / prices[-2] - 1) * 100 if len(prices) >= 2 else 0
    price_change_24h = (prices[-1] / prices[0] - 1) * 100 if len(prices) >= 2 else 0

    return {
        "pump_probability": round(float(final_score), 4),
        "risk_level": risk_level,
        "component_scores": {
            "random_walk": round(rw_score, 4),
            "hmm_regime": round(hmm_score, 4),
            "poisson_jumps": round(pj_score, 4),
            "volume_anomaly": round(volume_score, 4),
        },
        "random_walk": rw_result,
        "hmm_regime": hmm_result,
        "poisson_jumps": pj_result,
        "volume": {"ratio": round(volume_ratio, 3), "anomalous": volume_ratio > 3.0},
        "price_metrics": {
            "current_price": round(prices[-1], 6),
            "change_1h_pct": round(price_change_1h, 3),
            "change_24h_pct": round(price_change_24h, 3),
        },
    }


def simulate_path(scenario="pump", steps=60, S0=100.0):
    pj_model = PoissonJumpModel()
    if scenario == "pump":
        return pj_model.simulate_jump_path(S0=S0, mu=0.08, sigma=0.12, lam=0.3, mu_j=0.15, sigma_j=0.05, steps=steps)
    else:
        return pj_model.simulate_jump_path(S0=S0, mu=0.001, sigma=0.02, lam=0.02, mu_j=0.0, sigma_j=0.01, steps=steps)


async def fetch_coingecko(symbol, days=1):
    SYMBOL_MAP = {
        "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
        "DOGE": "dogecoin", "SHIB": "shiba-inu", "ADA": "cardano",
        "XRP": "ripple", "MATIC": "matic-network", "PEPE": "pepe", "BNB": "binancecoin",
    }
    coin_id = SYMBOL_MAP.get(symbol.upper(), symbol.lower())
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "hourly" if days <= 7 else "daily"}
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        prices = [p[1] for p in data["prices"]]
        volumes = [v[1] for v in data["total_volumes"]]
        return prices, volumes


def risk_color(risk):
    return {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}.get(risk, "⚪")


def plot_prices(prices, risk_level):
    color_map = {"LOW": "#22c55e", "MEDIUM": "#f59e0b", "HIGH": "#f97316", "CRITICAL": "#ef4444"}
    color = color_map.get(risk_level, "#6366f1")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=prices, mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=color.replace(")", ", 0.08)").replace("rgb", "rgba") if "rgb" in color else color + "15",
        name="Price"
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.1)", zeroline=False, tickfont=dict(size=10)),
        showlegend=False,
    )
    return fig


def plot_model_bars(scores):
    labels = ["Random Walk (GBM)", "HMM Regime", "Poisson Jumps", "Volume Spike"]
    values = [
        scores["random_walk"] * 100,
        scores["hmm_regime"] * 100,
        scores["poisson_jumps"] * 100,
        scores["volume_anomaly"] * 100,
    ]
    colors = []
    for v in values:
        if v > 75: colors.append("#ef4444")
        elif v > 50: colors.append("#f97316")
        elif v > 25: colors.append("#f59e0b")
        else: colors.append("#22c55e")

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        height=180,
        margin=dict(l=0, r=60, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 110], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=12)),
        showlegend=False,
    )
    return fig


# ── App Layout ────────────────────────────────────────────────────────────────

st.markdown("""
<h1 style='font-size:2rem;font-weight:700;margin-bottom:0'>
    🚨 Pump-and-Dump Detection Engine
</h1>
<p style='color:gray;margin-top:4px'>
    Real-time manipulation detection using Random Walk · HMM · Poisson Jump models
</p>
<hr>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    mode = st.radio("Data source", ["Live (CoinGecko)", "Simulate scenario"])

    if mode == "Live (CoinGecko)":
        symbol = st.selectbox("Token", ["PEPE", "SHIB", "DOGE", "BTC", "ETH", "SOL", "ADA", "XRP", "MATIC", "BNB"])
        days = st.slider("Days of data", 1, 30, 1)
        run = st.button("🔍 Analyze", use_container_width=True, type="primary")
    else:
        scenario = st.selectbox("Scenario", ["pump", "normal"])
        steps = st.slider("Price points", 30, 120, 60)
        S0 = st.number_input("Starting price ($)", value=100.0)
        run = st.button("▶ Run Simulation", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("""
    **Ensemble formula**
    ```
    Score = 0.30·RW 
          + 0.40·HMM 
          + 0.30·PJ 
          × vol_mult
    ```
    **Risk thresholds**
    - 🟢 < 35% → LOW
    - 🟡 35–65% → MEDIUM  
    - 🟠 65–80% → HIGH
    - 🔴 > 80% → CRITICAL
    """)

# ── Main Panel ────────────────────────────────────────────────────────────────

if run:
    with st.spinner("Running detection models..."):
        try:
            if mode == "Live (CoinGecko)":
                prices, volumes = asyncio.run(fetch_coingecko(symbol, days))
                title = f"{symbol} — Last {days} day(s)"
            else:
                prices = simulate_path(scenario=scenario, steps=steps, S0=S0)
                volumes = None
                title = f"Simulated {scenario.upper()} scenario ({steps} steps)"

            result = analyze(prices, volumes)

            if result is None:
                st.error("Not enough data points to analyze.")
            else:
                prob = result["pump_probability"]
                risk = result["risk_level"]
                scores = result["component_scores"]

                # ── Alert Banner ──────────────────────────────────────────
                if risk == "CRITICAL":
                    st.error(f"🔴 CRITICAL — Pump probability {prob*100:.1f}%. Very likely manipulation in progress. Do NOT buy.")
                elif risk == "HIGH":
                    st.warning(f"🟠 HIGH RISK — Pump probability {prob*100:.1f}%. Likely manipulation. Exercise caution.")
                elif risk == "MEDIUM":
                    st.info(f"🟡 MEDIUM — Pump probability {prob*100:.1f}%. Elevated signal. Monitor closely.")
                else:
                    st.success(f"🟢 LOW — Pump probability {prob*100:.1f}%. Normal market conditions.")

                # ── Top Metrics ───────────────────────────────────────────
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Pump Probability", f"{prob*100:.1f}%")
                with col2:
                    st.metric("Risk Level", f"{risk_color(risk)} {risk}")
                with col3:
                    change_1h = result["price_metrics"]["change_1h_pct"]
                    st.metric("Price Δ (1h)", f"{change_1h:+.2f}%")
                with col4:
                    change_24h = result["price_metrics"]["change_24h_pct"]
                    st.metric("Price Δ (total)", f"{change_24h:+.2f}%")

                st.markdown(f"### {title}")

                # ── Charts ────────────────────────────────────────────────
                col_left, col_right = st.columns([3, 2])

                with col_left:
                    st.markdown("**Price action**")
                    st.plotly_chart(plot_prices(prices, risk), use_container_width=True)

                    # Ensemble score bar
                    st.markdown("**Ensemble score**")
                    bar_color = "#ef4444" if prob > 0.75 else "#f97316" if prob > 0.50 else "#f59e0b" if prob > 0.35 else "#22c55e"
                    st.markdown(f"""
                    <div style="background:#1e1e2e;border-radius:8px;overflow:hidden;height:12px;margin-bottom:4px">
                        <div style="width:{prob*100:.1f}%;height:100%;background:{bar_color};border-radius:8px;transition:width 0.5s"></div>
                    </div>
                    <p style="font-size:12px;color:gray">{prob*100:.1f}% — {risk}</p>
                    """, unsafe_allow_html=True)

                with col_right:
                    st.markdown("**Model breakdown**")
                    st.plotly_chart(plot_model_bars(scores), use_container_width=True)

                    # HMM Regime
                    regime = result["hmm_regime"]["regime"]
                    regime_icon = {"normal": "😴", "trending": "📈", "pump": "🚀", "unknown": "❓"}.get(regime, "❓")
                    st.markdown(f"**HMM Regime:** {regime_icon} `{regime.upper()}`")

                    # Volume
                    vol_ratio = result["volume"]["ratio"]
                    vol_flag = "🚨 Anomalous" if result["volume"]["anomalous"] else "✅ Normal"
                    st.markdown(f"**Volume ratio:** `{vol_ratio:.2f}x` {vol_flag}")

                    # Jump
                    jump_intensity = result["poisson_jumps"]["jump_intensity"]
                    recent_jump = result["poisson_jumps"]["recent_jump"]
                    st.markdown(f"**Jump intensity (λ):** `{jump_intensity:.4f}`")
                    if recent_jump:
                        st.markdown("⚡ **Jump event detected in latest period**")

                # ── Statistical Detail ────────────────────────────────────
                with st.expander("📊 Statistical detail"):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.markdown("**Random Walk (GBM)**")
                        rw = result["random_walk"]
                        st.code(f"z-score:  {rw['z_score']:.4f}\nmu:       {rw['mu']:.6f}\nsigma:    {rw['sigma']:.6f}\nabnormal: {rw['is_abnormal']}")
                    with col_b:
                        st.markdown("**HMM Regime**")
                        hmm = result["hmm_regime"]
                        probs = hmm["state_probs"]
                        st.code(f"regime:   {hmm['regime']}\nnormal:   {probs[0]:.4f}\ntrending: {probs[1]:.4f}\npump:     {probs[2]:.4f}")
                    with col_c:
                        st.markdown("**Poisson Jumps**")
                        pj = result["poisson_jumps"]
                        st.code(f"lambda:   {pj['jump_intensity']:.4f}\npos_ratio:{pj['positive_jump_ratio']:.4f}\nn_jumps:  {pj['n_jumps']}\nrecent:   {pj['recent_jump']}")

        except Exception as e:
            st.error(f"Error: {str(e)}")
            if mode == "Live (CoinGecko)":
                st.info("CoinGecko may be rate-limiting. Try the Simulate scenario mode instead.")

else:
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;color:gray">
        <h2>👈 Configure and run an analysis</h2>
        <p>Select a token or simulation scenario from the sidebar, then click Analyze.</p>
        <br>
        <p><b>Models used:</b></p>
        <p>📉 Geometric Brownian Motion (z-score abnormality)</p>
        <p>🔄 Hidden Markov Model (regime detection)</p>
        <p>⚡ Poisson Jump Process (Merton model)</p>
    </div>
    """, unsafe_allow_html=True)
