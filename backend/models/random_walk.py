"""
Stochastic Random Walk Model
-----------------------------
Models normal crypto price movement as a Geometric Brownian Motion (GBM):
    dS = μ·S·dt + σ·S·dW
where:
    S = price
    μ = drift (expected return)
    σ = volatility
    dW = Wiener process (Brownian motion increment)

We use this to define the EXPECTED distribution of price moves.
Any move falling far outside this distribution is flagged as anomalous.
"""

import numpy as np


class RandomWalkModel:
    def __init__(self, window: int = 20):
        """
        window: rolling window (in periods) to estimate μ and σ
        """
        self.window = window

    def compute_log_returns(self, prices: list[float]) -> np.ndarray:
        """Convert price series to log returns: r_t = ln(S_t / S_{t-1})"""
        prices = np.array(prices, dtype=float)
        log_returns = np.diff(np.log(prices))
        return log_returns

    def estimate_params(self, log_returns: np.ndarray):
        """
        Estimate drift (μ) and volatility (σ) from log return history.
        Uses rolling window for adaptivity.
        """
        window_data = log_returns[-self.window:] if len(log_returns) >= self.window else log_returns
        mu = float(np.mean(window_data))
        sigma = float(np.std(window_data)) + 1e-9  # avoid division by zero
        return mu, sigma

    def z_score(self, log_return: float, mu: float, sigma: float) -> float:
        """
        Compute z-score of a single log return under N(μ, σ²).
        z = (r - μ) / σ
        High absolute z-score = abnormal move.
        """
        return (log_return - mu) / sigma

    def abnormality_score(self, prices: list[float]) -> dict:
        """
        Main API: given price history, return:
        - z_score of latest return
        - rolling mu, sigma
        - whether the latest move is abnormal (|z| > threshold)
        """
        if len(prices) < 3:
            return {"z_score": 0.0, "mu": 0.0, "sigma": 0.0, "is_abnormal": False}

        log_returns = self.compute_log_returns(prices)
        mu, sigma = self.estimate_params(log_returns[:-1])  # exclude latest for fair eval
        latest_return = log_returns[-1]
        z = self.z_score(latest_return, mu, sigma)

        return {
            "z_score": round(z, 4),
            "mu": round(mu, 6),
            "sigma": round(sigma, 6),
            "latest_log_return": round(latest_return, 6),
            "is_abnormal": abs(z) > 2.5,  # 99% confidence band
        }

    def simulate(self, S0: float, mu: float, sigma: float, steps: int = 50) -> list[float]:
        """
        Simulate future price path under GBM (for visualization).
        S_{t+1} = S_t * exp((μ - σ²/2)*dt + σ*√dt*Z)
        """
        dt = 1  # 1 period
        prices = [S0]
        for _ in range(steps):
            Z = np.random.standard_normal()
            S_next = prices[-1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
            prices.append(float(S_next))
        return prices