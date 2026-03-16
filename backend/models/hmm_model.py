"""
Hidden Markov Model (Regime Switching)
---------------------------------------
Crypto markets operate in distinct REGIMES (states):
    State 0: Normal / Bear  → low volatility, small moves
    State 1: Bull trending  → moderate positive drift
    State 2: Pump regime    → very high volatility, large positive moves (ANOMALOUS)

We model the OBSERVABLE = log returns as Gaussian emissions from each hidden state.
The Viterbi algorithm decodes the most likely state sequence.
The forward algorithm gives posterior state probabilities at each step.

Key Markov property: P(state_t | state_{t-1}, state_{t-2}, ...) = P(state_t | state_{t-1})
→ Only the previous state matters, which allows efficient inference.
"""

import numpy as np
from hmmlearn import hmm


class HMMRegimeModel:
    def __init__(self, n_states: int = 3):
        """
        n_states: number of hidden market regimes
        We use a Gaussian HMM (observations modeled as Gaussians per state)
        """
        self.n_states = n_states
        self.model = None
        self.is_fitted = False

        # State labels for interpretation
        self.state_labels = {
            0: "normal",
            1: "trending",
            2: "pump",
        }

    def fit(self, log_returns: np.ndarray):
        """
        Fit the HMM to observed log return sequence.
        Uses Baum-Welch (EM) algorithm to estimate:
            - Transition matrix A (state → state probs)
            - Emission parameters μ_i, σ_i per state
            - Initial state probabilities π
        Minimum 30 data points recommended.
        """
        if len(log_returns) < 20:
            return False

        X = log_returns.reshape(-1, 1)  # hmmlearn expects (T, n_features)

        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=200,
            random_state=42,
            tol=1e-4,
        )
        try:
            self.model.fit(X)
            self.is_fitted = True

            # Sort states by volatility (std) ascending so state 0=calm, 2=pump
            stds = [np.sqrt(self.model.covars_[i][0][0]) for i in range(self.n_states)]
            sort_order = np.argsort(stds)
            self._remap = {old: new for new, old in enumerate(sort_order)}
            return True
        except Exception:
            self.is_fitted = False
            return False

    def predict_state(self, log_returns: np.ndarray) -> dict:
        """
        Given log return history, decode the current hidden state.
        Returns: current_state (int), state_probs (list), regime_label (str)
        """
        if not self.is_fitted or len(log_returns) < 5:
            return {
                "current_state": 0,
                "state_probs": [1.0 / self.n_states] * self.n_states,
                "regime": "unknown",
                "pump_probability": 0.0,
            }

        X = log_returns.reshape(-1, 1)

        # Forward algorithm: posterior state probabilities
        try:
            log_prob, posteriors = self.model.score_samples(X)
            current_posteriors = posteriors[-1].tolist()

            # Viterbi: most likely state sequence
            _, states = self.model.decode(X, algorithm="viterbi")
            current_state_raw = int(states[-1])
            current_state = self._remap.get(current_state_raw, current_state_raw)

            # Remap posteriors too
            remapped_probs = [0.0] * self.n_states
            for i, p in enumerate(current_posteriors):
                remapped_probs[self._remap.get(i, i)] += p

            pump_prob = remapped_probs[2]  # state 2 = pump after sorting

            return {
                "current_state": current_state,
                "state_probs": [round(p, 4) for p in remapped_probs],
                "regime": self.state_labels.get(current_state, "unknown"),
                "pump_probability": round(float(pump_prob), 4),
            }
        except Exception:
            return {
                "current_state": 0,
                "state_probs": [1.0 / self.n_states] * self.n_states,
                "regime": "normal",
                "pump_probability": 0.0,
            }

    def get_transition_matrix(self) -> list:
        """Return the fitted state transition matrix."""
        if self.is_fitted:
            return self.model.transmat_.tolist()
        return []

    def get_emission_params(self) -> list:
        """Return mean and std of each state's emission distribution."""
        if not self.is_fitted:
            return []
        result = []
        for i in range(self.n_states):
            mean = float(self.model.means_[i][0])
            std = float(np.sqrt(self.model.covars_[i][0][0]))
            label = self.state_labels.get(self._remap.get(i, i), "state")
            result.append({"state": i, "label": label, "mean": round(mean, 6), "std": round(std, 6)})
        return result   