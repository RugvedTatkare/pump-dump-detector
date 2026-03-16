"""
Alert Manager
--------------
Manages alert generation, deduplication, and history.
Alerts are triggered when pump_probability crosses risk thresholds.
"""

from datetime import datetime, timezone
from collections import deque
import json


class AlertManager:
    def __init__(self, max_history: int = 100):
        self.history = deque(maxlen=max_history)
        self._last_alert_level = {}  # symbol → last alerted level

    def check_and_generate(self, symbol: str, analysis: dict) -> dict | None:
        """
        Generate an alert if conditions warrant it.
        Deduplicates: won't re-alert the same level for the same symbol
        until the level drops and re-triggers.
        """
        score = analysis.get("pump_probability", 0)
        risk = analysis.get("risk_level", "LOW")

        # Only alert at MEDIUM or above
        if risk == "LOW":
            self._last_alert_level[symbol] = "LOW"
            return None

        # Deduplicate: only alert if this is a new or escalated level
        prev_level = self._last_alert_level.get(symbol, "LOW")
        level_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        if level_order.get(risk, 0) <= level_order.get(prev_level, 0):
            return None  # same or lower, skip

        # Generate alert
        alert = {
            "id": f"{symbol}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "risk_level": risk,
            "pump_probability": score,
            "price": analysis.get("price_metrics", {}).get("current_price", 0),
            "change_1h": analysis.get("price_metrics", {}).get("change_1h_pct", 0),
            "change_24h": analysis.get("price_metrics", {}).get("change_24h_pct", 0),
            "regime": analysis.get("hmm_regime", {}).get("regime", "unknown"),
            "recent_jump": analysis.get("poisson_jumps", {}).get("recent_jump", False),
            "volume_anomalous": analysis.get("volume", {}).get("anomalous", False),
            "message": self._compose_message(symbol, risk, score, analysis),
        }

        self.history.appendleft(alert)
        self._last_alert_level[symbol] = risk
        return alert

    def _compose_message(self, symbol: str, risk: str, score: float, analysis: dict) -> str:
        price = analysis.get("price_metrics", {}).get("current_price", 0)
        change = analysis.get("price_metrics", {}).get("change_1h_pct", 0)
        regime = analysis.get("hmm_regime", {}).get("regime", "unknown")
        jump = analysis.get("poisson_jumps", {}).get("recent_jump", False)

        msg = f"[{risk}] {symbol} — Pump probability {score*100:.1f}%. "
        msg += f"Price: ${price:.4f} ({change:+.2f}% 1h). "
        msg += f"Regime: {regime}. "
        if jump:
            msg += "Jump event detected. "
        if analysis.get("volume", {}).get("anomalous"):
            msg += "Volume spike anomaly."
        return msg.strip()

    def get_history(self, limit: int = 50, symbol: str = None) -> list:
        alerts = list(self.history)
        if symbol:
            alerts = [a for a in alerts if a["symbol"] == symbol]
        return alerts[:limit]

    def get_stats(self) -> dict:
        alerts = list(self.history)
        by_level = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        for a in alerts:
            by_level[a.get("risk_level", "LOW")] += 1
        return {
            "total_alerts": len(alerts),
            "by_risk_level": by_level,
            "symbols_tracked": list(set(a["symbol"] for a in alerts)),
        }