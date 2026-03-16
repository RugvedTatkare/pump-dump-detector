"""
Poisson Jump Process Model
---------------------------
Merton's Jump-Diffusion model augments GBM with Poisson jump events:
    dS = μ·S·dt + σ·S·dW + J·S·dN(λ)
where:
    dN(λ) = Poisson process with intensity λ (expected jumps per period)
    J      = jump size ~ N(μ_J, σ_J²)

Key insight for pump detection:
    Under NORMAL conditions, large price jumps are RARE (Poisson events with low λ).
    During a PUMP, jumps become FREQUENT and consistently POSITIVE.
    We estimate the JUMP INTENSITY λ and JUMP DIRECTION from recent data.
    High λ with positive bias → strong pump signal.

We separate the price path into:
    1. Diffusion component (normal GBM)
    2. Jump component (deviations exceeding 3σ threshold)
"""

import numpy as np
from scipy.stats import norm, poisson


class PoissonJumpModel:
    def __init__(self, jump_threshold_sigma: float = 2.5):
        """
        jump_threshold_sigma: how many σ away from GBM trend
                              a return must be to count as a 'jump'
        """
        self.jump_threshold_sigma = jump_threshold_sigma

    def detect_jumps(self, log_returns: np.ndarray) -> dict:
        """
        Identify jump events in the return series.
        A jump is a return whose |z-score| > jump_threshold_sigma.

        Returns:
            jumps: boolean array of which periods had jumps
            jump_sizes: sizes of detected jumps
            jump_intensity: estimated λ (jumps per period)
            positive_jump_ratio: fraction of jumps that were UP
        """
        if len(log_returns) < 5:
            return {
                "jump_intensity": 0.0,
                "positive_jump_ratio": 0.5,
                "n_jumps": 0,
                "jump_sizes": [],
                "recent_jump": False,
            }

        # Separate diffusion from jump component
        mu = np.mean(log_returns)
        sigma = np.std(log_returns) + 1e-9
        z_scores = (log_returns - mu) / sigma

        jump_mask = np.abs(z_scores) > self.jump_threshold_sigma
        jump_indices = np.where(jump_mask)[0]
        jump_sizes = log_returns[jump_mask].tolist()
        n_jumps = int(np.sum(jump_mask))

        # Estimate jump intensity λ = E[dN] = jumps per period
        T = len(log_returns)
        lam = n_jumps / T if T > 0 else 0.0

        # Direction bias: what fraction were positive (pump) vs negative (dump)
        positive_jumps = sum(1 for j in jump_sizes if j > 0)
        pos_ratio = positive_jumps / n_jumps if n_jumps > 0 else 0.5

        # Was the most recent period a jump?
        recent_jump = bool(jump_mask[-1]) if len(jump_mask) > 0 else False

        return {
            "jump_intensity": round(lam, 4),
            "positive_jump_ratio": round(pos_ratio, 4),
            "n_jumps": n_jumps,
            "jump_sizes": [round(j, 6) for j in jump_sizes[-10:]],  # last 10
            "recent_jump": recent_jump,
            "recent_z_score": round(float(z_scores[-1]), 3),
        }

    def pump_score_from_jumps(
        self, jump_intensity: float, positive_jump_ratio: float, recent_jump: bool
    ) -> float:
        """
        Combine jump statistics into a 0-1 pump probability contribution.

        Logic:
            - High λ (many jumps) = abnormal market activity
            - High positive ratio = directional (pump, not random volatility)
            - Recent jump = current event (not historical)

        Score formula is heuristic but interpretable for the presentation.
        """
        # Intensity score: sigmoid-scaled
        intensity_score = 1 - np.exp(-5 * jump_intensity)  # saturates near 1 at λ≈0.6+

        # Direction score: distance from 0.5 (neutral), scaled [0,1]
        direction_score = (positive_jump_ratio - 0.5) * 2  # −1 to +1; >0 means pump bias
        direction_score = max(0.0, direction_score)  # only care about upward bias

        # Recency bonus
        recency_bonus = 0.15 if recent_jump else 0.0

        score = 0.5 * intensity_score + 0.35 * direction_score + recency_bonus
        return round(min(1.0, max(0.0, score)), 4)

    def poisson_exceedance_probability(self, observed_jumps: int, T: int, lambda_baseline: float) -> float:
        """
        What is the probability of observing >= observed_jumps in T periods
        if the true jump intensity is lambda_baseline (e.g., historical average)?

        P(X >= k) = 1 - P(X <= k-1)    where X ~ Poisson(λ·T)

        A very small p-value means the current jump rate is statistically
        unlikely under normal conditions → manipulation signal.
        """
        if observed_jumps == 0:
            return 1.0
        expected = lambda_baseline * T
        if expected <= 0:
            return 0.0
        p_value = 1.0 - poisson.cdf(observed_jumps - 1, expected)
        return round(float(p_value), 6)

    def simulate_jump_path(
        self,
        S0: float,
        mu: float,
        sigma: float,
        lam: float,
        mu_j: float,
        sigma_j: float,
        steps: int = 50,
    ) -> list[float]:
        """
        Simulate a Merton jump-diffusion path (for visualization):
            S_{t+1} = S_t * exp(diffusion) * (1 + J * Poisson_event)
        """
        prices = [S0]
        dt = 1
        for _ in range(steps):
            # Diffusion
            diffusion = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal()
            # Jump
            n_events = np.random.poisson(lam * dt)
            jump = sum(np.random.normal(mu_j, sigma_j) for _ in range(n_events))
            prices.append(float(prices[-1] * np.exp(diffusion + jump)))
        return prices