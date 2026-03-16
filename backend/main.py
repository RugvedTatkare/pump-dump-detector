"""
Cryptocurrency Pump-and-Dump Detection API
==========================================
FastAPI backend exposing:
    GET  /api/analyze/{symbol}      → Run full detection on a symbol
    POST /api/analyze/custom        → Analyze user-provided price data
    GET  /api/alerts                → Retrieve alert history
    GET  /api/simulate              → Simulate normal vs pump path
    GET  /api/historical/{symbol}   → Historical manipulation scan
    GET  /health                    → Health check
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import httpx
import asyncio
import os
from datetime import datetime, timezone

from detector import PumpDetector
from alert_manager import AlertManager
from models.random_walk import RandomWalkModel
from models.poisson_jumps import PoissonJumpModel

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Crypto Pump-and-Dump Detection Engine",
    description="Stochastic model using Random Walk, HMM, and Poisson Jump processes",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global State ──────────────────────────────────────────────────────────────
detector = PumpDetector()
alert_manager = AlertManager()
rw_model = RandomWalkModel()
pj_model = PoissonJumpModel()

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# Map user-friendly symbols to CoinGecko IDs
SYMBOL_MAP = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "DOGE": "dogecoin",
    "SHIB": "shiba-inu",
    "ADA": "cardano",
    "XRP": "ripple",
    "MATIC": "matic-network",
    "PEPE": "pepe",
    "BNB": "binancecoin",
}


# ── Data Fetching ─────────────────────────────────────────────────────────────

async def fetch_prices(symbol: str, days: int = 1) -> tuple[list[float], list[float]]:
    """Fetch OHLCV from CoinGecko. Returns (prices, volumes)."""
    coin_id = SYMBOL_MAP.get(symbol.upper(), symbol.lower())
    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "hourly" if days <= 7 else "daily"}

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            prices = [p[1] for p in data["prices"]]
            volumes = [v[1] for v in data["total_volumes"]]
            return prices, volumes
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch price data: {str(e)}")


# ── Request / Response Models ─────────────────────────────────────────────────

class CustomAnalysisRequest(BaseModel):
    prices: list[float]
    volumes: Optional[list[float]] = None
    symbol: Optional[str] = "CUSTOM"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/api/analyze/{symbol}")
async def analyze_symbol(
    symbol: str,
    days: int = Query(default=1, ge=1, le=90, description="Number of days of data"),
):
    """
    Fetch live price data for {symbol} and run the full detection pipeline.
    Returns pump probability, risk level, and per-model breakdown.
    """
    symbol = symbol.upper()
    prices, volumes = await fetch_prices(symbol, days=days)

    if len(prices) < 5:
        raise HTTPException(status_code=422, detail="Insufficient price data returned")

    detector.update(prices, volumes)
    result = detector.analyze(prices, volumes)
    result["symbol"] = symbol
    result["fetched_at"] = datetime.now(timezone.utc).isoformat()

    # Check for alerts
    alert = alert_manager.check_and_generate(symbol, result)
    result["alert_generated"] = alert is not None
    if alert:
        result["alert"] = alert

    return result


@app.post("/api/analyze/custom")
def analyze_custom(req: CustomAnalysisRequest):
    """
    Analyze user-supplied price (and optional volume) arrays.
    Useful for backtesting or demo purposes.
    """
    if len(req.prices) < 5:
        raise HTTPException(status_code=422, detail="Need at least 5 price points")

    detector.update(req.prices, req.volumes)
    result = detector.analyze(req.prices, req.volumes)
    result["symbol"] = req.symbol
    result["fetched_at"] = datetime.now(timezone.utc).isoformat()

    alert = alert_manager.check_and_generate(req.symbol, result)
    result["alert_generated"] = alert is not None
    return result


@app.get("/api/alerts")
def get_alerts(limit: int = Query(default=20, le=100), symbol: Optional[str] = None):
    """Return recent alert history, optionally filtered by symbol."""
    return {
        "alerts": alert_manager.get_history(limit=limit, symbol=symbol),
        "stats": alert_manager.get_stats(),
    }


@app.get("/api/simulate")
def simulate(
    scenario: str = Query(default="pump", description="'normal' or 'pump'"),
    steps: int = Query(default=60, ge=10, le=200),
    S0: float = Query(default=100.0),
):
    """
    Simulate a price path and run detection on it.
    Useful for demo/education: see how scores respond to known pump vs normal.
    """
    if scenario == "pump":
        # Simulate a pump: high vol + strong positive drift + frequent jumps
        path = pj_model.simulate_jump_path(
            S0=S0, mu=0.08, sigma=0.12, lam=0.3, mu_j=0.15, sigma_j=0.05, steps=steps
        )
    else:
        # Normal: low vol, near-zero drift, rare small jumps
        path = pj_model.simulate_jump_path(
            S0=S0, mu=0.001, sigma=0.02, lam=0.02, mu_j=0.0, sigma_j=0.01, steps=steps
        )

    detector.update(path)
    result = detector.analyze(path)
    result["simulated_path"] = [round(p, 4) for p in path]
    result["scenario"] = scenario
    return result


@app.get("/api/historical/{symbol}")
async def historical_scan(
    symbol: str,
    days: int = Query(default=30, ge=7, le=365),
):
    """
    Scan historical data and flag time windows with high pump probability.
    Returns a time series of [timestamp, pump_probability] for charting.
    """
    symbol = symbol.upper()
    prices, volumes = await fetch_prices(symbol, days=days)

    if len(prices) < 20:
        raise HTTPException(status_code=422, detail="Not enough historical data")

    # Rolling window analysis
    window = 24  # hours
    results = []
    max_score = 0.0
    max_idx = 0

    for i in range(window, len(prices)):
        window_prices = prices[max(0, i - window): i + 1]
        window_volumes = volumes[max(0, i - window): i + 1] if volumes else None
        r = detector.analyze(window_prices, window_volumes)
        score = r["pump_probability"]
        results.append({"index": i, "score": score, "risk": r["risk_level"]})
        if score > max_score:
            max_score = score
            max_idx = i

    suspicious_windows = [r for r in results if r["score"] > 0.35]

    return {
        "symbol": symbol,
        "days": days,
        "total_windows": len(results),
        "max_pump_probability": round(max_score, 4),
        "suspicious_windows_count": len(suspicious_windows),
        "timeline": results,
        "most_suspicious_index": max_idx,
    }


@app.get("/api/supported-coins")
def supported_coins():
    return {"coins": list(SYMBOL_MAP.keys())}