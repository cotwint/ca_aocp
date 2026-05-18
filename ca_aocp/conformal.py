"""
Weighted Conformal Calibration
================================
Implements equations (3.5)–(3.7) from the paper.

Given normalised weights w̃_{t,i} and past conformity scores S_{1:t-1},
we form the weighted empirical CDF F̂^w_t and take the
(1 − α_t)-quantile as the conformal radius q_t.
"""

from __future__ import annotations
import numpy as np


def weighted_quantile(
    scores: np.ndarray,
    weights: np.ndarray,
    level: float,
) -> float:
    """Weighted (1 − level)-quantile of the empirical score distribution.

    Implements Q_{1−α_t}(F̂^w_t) from equation (3.6):

        q_t = inf{ s : F̂^w_t(s) ≥ 1 − α_t }

    Parameters
    ----------
    scores : np.ndarray, shape (n,)
        Historical conformity scores S_1, …, S_{t-1}.
    weights : np.ndarray, shape (n,)
        Normalised importance weights w̃_{t,1}, …, w̃_{t,t-1}.
    level : float
        Current effective miscoverage level α_t ∈ (0, 1).
        We seek the (1 − α_t) quantile.

    Returns
    -------
    q : float
        Conformal radius q_t.
    """
    assert len(scores) == len(weights), "scores and weights must have the same length."
    assert 0.0 < level < 1.0, "level must be in (0, 1)."

    # Sort by score value
    order = np.argsort(scores)
    s_sorted = scores[order]
    w_sorted = weights[order]

    # Weighted empirical CDF: F̂^w_t(s_k) = Σ_{i: S_i ≤ s_k} w̃_i
    cdf = np.cumsum(w_sorted)

    # Find smallest s_k such that F̂^w_t(s_k) ≥ 1 − α_t
    target = 1.0 - level
    idx = np.searchsorted(cdf, target, side="left")
    idx = min(idx, len(s_sorted) - 1)
    return float(s_sorted[idx])


def prediction_interval(
    prediction: float,
    radius: float,
) -> tuple[float, float]:
    """Symmetric prediction interval [ŷ − q, ŷ + q].

    Parameters
    ----------
    prediction : float
        Point estimate ŷ = f̂_t(X_t).
    radius : float
        Conformal radius q_t.

    Returns
    -------
    lower, upper : float
        Prediction interval endpoints.
    """
    return prediction - radius, prediction + radius


def coverage_indicator(y_true: float, lower: float, upper: float) -> int:
    """Return 1 if y_true ∈ [lower, upper], else 0."""
    return int(lower <= y_true <= upper)
