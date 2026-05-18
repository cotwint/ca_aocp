"""
Changepoint-Aware Importance Weights
=====================================
Implements equation (3.2) from the paper:

    w_{t,i} = π_{t,i} · exp(−λ(t − i))

with normalisation and effective-sample-size computation.
"""

from __future__ import annotations
import numpy as np


def compute_weights(
    pi: np.ndarray,
    decay: float,
    t: int,
) -> np.ndarray:
    """Compute normalised changepoint-aware importance weights.

    Parameters
    ----------
    pi : np.ndarray, shape (t-1,)
        Regime relevance scores π_{t,i} for i = 1, …, t−1.
        pi[i-1] = π_{t,i} = P(observation i is from current regime).
    decay : float
        Forgetting parameter λ ≥ 0.  λ=0 uses pure BOCPD weights.
    t : int
        Current time step (1-indexed).

    Returns
    -------
    w_tilde : np.ndarray, shape (t-1,)
        Normalised weights summing to 1.
        w_tilde[i-1] is the weight for historical observation i.
    """
    n = len(pi)  # = t - 1
    assert n > 0, "No historical observations yet."

    # Lags: observation i was t−i steps ago; i=1,...,t-1 → lags t-1,...,1
    lags = np.arange(n, 0, -1, dtype=float)  # lags[i-1] = t - i

    # Unnormalised weights: w_{t,i} = π_{t,i} · exp(−λ · (t−i))
    log_decay = -decay * lags
    log_w = np.log(np.clip(pi, 1e-300, None)) + log_decay
    log_w -= np.max(log_w)  # numerical stability before exp
    w = np.exp(log_w)

    total = w.sum()
    if total < 1e-300:
        # Degenerate: fall back to uniform
        return np.ones(n) / n
    return w / total


def effective_sample_size(w_tilde: np.ndarray) -> float:
    """N_eff = (Σ w̃²)^{-1}  (equation 3.4 in the paper)."""
    return 1.0 / float(np.dot(w_tilde, w_tilde))


def pre_change_weight_mass(w_tilde: np.ndarray, tau: int, t: int) -> float:
    """Total weight on pre-change observations Σ_{i≤τ} w̃_{t,i}.

    Useful for monitoring contamination after a known changepoint τ.
    """
    if tau <= 0 or tau >= t - 1:
        return float(np.nan)
    # w_tilde[i-1] is for observation i; pre-change means i ≤ τ
    return float(w_tilde[:tau].sum())
