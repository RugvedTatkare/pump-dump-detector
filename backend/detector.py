"""
Core Pump-and-Dump Detection Engine
-------------------------------------
Combines the three stochastic models into a unified pump probability score:

    Final Score = w1 * RandomWalk_score + w2 * HMM_score + w3 * Poisson_score

Where:
    RandomWalk_score: normalized |z-score| of latest return under GBM
    HMM_score:        posterior probability of 'pump' hidden state
    Poisson_score:    jump-derived pump probability

Volume anomaly detection is also included as a multiplier:
    volume_multiplier = clamp(volume_ratio / normal_volume_ratio, 1.0, 3.0)

This gives a final_score ∈ [0, 1] with interpretable thresholds:
    < 0.35  → LOW risk (normal market)
    0.35-0.65 → MEDIUM risk (watch)
    0.65-0.80 → HIGH risk (likely manipulation)
    > 0.80  → CRITICAL (very likely pump-and-dump)
"""

import numpy as np
from typing import Optional
from .models.random_walk import RandomWalkModel
from .models.hmm_model import HMMRegimeModel
from .models.poisson_jumps import PoissonJumpModel


WEIGHTS = {"random_walk": 0.30, "hmm": 0.40, "poisson": 0.30}

RISK_THRESHOLDS = {
    "low": 0.35,
    "medium": 0.65,
    "high": 0.80,
}


class PumpDetector:
    def __init__(self):
        self.rw_model = RandomWalkModel(window=20)
        self.hmm_model = HMMRegimeModel(n_states=3)
        self.pj_model = PoissonJumpModel(jump_threshold_sigma=2.5)
        self.hmm_fitted = False
        self._price_buffer = []
        self._volume_buffer = []

    def update(self, prices: list[float], volumes: Optional[list[float]] = None):
        """
        Update internal buffers. Refit HMM if we have enough data.
        Call this whenever new data arrives.
        """
        self._price_buffer = prices
        self._volume_buffer = volumes or []

        if len(prices) >= 30:
            log_returns = self.rw_model.compute_log_returns(prices)
            self.hmm_fitted = self.hmm_model.fit(log_returns)

    def analyze(self, prices: list[float], volumes: Optional[list[float]] = None) -> dict:
        """
        Full analysis pipeline. Returns comprehensive detection result.
        """
        if len(prices) < 5:
            return self._empty_result()

        log_returns = self.rw_model.compute_log_returns(prices)

        # ── 1. Random Walk Analysis ──────────────────────────────────────────
        rw_result = self.rw_model.abnormality_score(prices)
        # Normalize |z| to [0,1] using sigmoid: score = 1 - exp(-|z|/3)
        rw_score = float(1 - np.exp(-abs(rw_result["z_score"]) / 3.0))

        # ── 2. HMM Regime Detection ──────────────────────────────────────────
        if not self.hmm_fitted and len(log_returns) >= 20:
            self.hmm_fitted = self.hmm_model.fit(log_returns)

        hmm_result = self.hmm_model.predict_state(log_returns)
        hmm_score = hmm_result["pump_probability"]

        # ── 3. Poisson Jump Detection ────────────────────────────────────────
        pj_result = self.pj_model.detect_jumps(log_returns)
        pj_score = self.pj_model.pump_score_from_jumps(
            pj_result["jump_intensity"],
            pj_result["positive_jump_ratio"],
            pj_result["recent_jump"],
        )

        # ── 4. Volume Anomaly ────────────────────────────────────────────────
        volume_score = 0.0
        volume_ratio = 1.0
        if volumes and len(volumes) >= 5:
            baseline_vol = float(np.mean(volumes[:-1]))
            current_vol = float(volumes[-1])
            if baseline_vol > 0:
                volume_ratio = current_vol / baseline_vol
                volume_score = min(1.0, (volume_ratio - 1.0) / 5.0)  # 0 at 1x, 1 at 6x

        # ── 5. Weighted Ensemble ─────────────────────────────────────────────
        base_score = (
            WEIGHTS["random_walk"] * rw_score
            + WEIGHTS["hmm"] * hmm_score
            + WEIGHTS["poisson"] * pj_score
        )

        # Volume multiplier: if volume also spikes, boost score
        volume_multiplier = 1.0 + 0.3 * volume_score  # max 1.3x
        final_score = min(1.0, base_score * volume_multiplier)

        # ── 6. Risk Classification ───────────────────────────────────────────
        if final_score < RISK_THRESHOLDS["low"]:
            risk_level = "LOW"
        elif final_score < RISK_THRESHOLDS["medium"]:
            risk_level = "MEDIUM"
        elif final_score < RISK_THRESHOLDS["high"]:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"

        # ── 7. Price Change Metrics ──────────────────────────────────────────
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
            "volume": {
                "ratio": round(volume_ratio, 3),
                "anomalous": volume_ratio > 3.0,
            },
            "price_metrics": {
                "current_price": round(prices[-1], 6),
                "change_1h_pct": round(price_change_1h, 3),
                "change_24h_pct": round(price_change_24h, 3),
            },
            "data_points_used": len(prices),
        }

    def _empty_result(self) -> dict:
        return {
            "pump_probability": 0.0,
            "risk_level": "LOW",
            "component_scores": {"random_walk": 0, "hmm_regime": 0, "poisson_jumps": 0, "volume_anomaly": 0},
            "message": "Insufficient data (need at least 5 price points)",
        }