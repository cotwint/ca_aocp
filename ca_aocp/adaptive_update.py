"""
Adaptive Miscoverage Level Update
====================================
Implements equations (3.8)–(3.9) from the paper.

The update rule is:
    ℓ_t = σ((S_t − q_t) / τ)          [smooth surrogate]
    α_{t+1} = Π_{[α_min, α_max]}(α_t + η(α − ℓ_t))

where σ is the logistic function and τ > 0 is a temperature.
As τ → 0, ℓ_t → 1{S_t > q_t}, recovering binary ACI.
"""

from __future__ import annotations
import numpy as np


def logistic(x: float | np.ndarray) -> float | np.ndarray:
    """Numerically stable logistic σ(x) = 1 / (1 + e^{-x})."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def smooth_surrogate(score: float, radius: float, temperature: float) -> float:
    """Smooth miscoverage surrogate ℓ_t = σ((S_t − q_t) / τ).

    Parameters
    ----------
    score : float
        Current conformity score S_t.
    radius : float
        Conformal radius q_t.
    temperature : float
        Smoothing temperature τ > 0.  Small τ → binary indicator.

    Returns
    -------
    ell : float
        Smooth surrogate ∈ (0, 1).
    """
    assert temperature > 0, "temperature must be positive."
    return float(logistic((score - radius) / temperature))


def adaptive_update(
    alpha_t: float,
    ell_t: float,
    alpha_target: float,
    eta: float,
    alpha_min: float,
    alpha_max: float,
) -> float:
    """One step of the projected gradient update on α_t.

    α_{t+1} = Π_{[α_min, α_max]}(α_t + η(α − ℓ_t))

    Parameters
    ----------
    alpha_t : float
        Current effective miscoverage level.
    ell_t : float
        Smooth surrogate loss ∈ (0, 1).
    alpha_target : float
        Nominal miscoverage level α.
    eta : float
        Step size η > 0.
    alpha_min, alpha_max : float
        Feasibility bounds.

    Returns
    -------
    alpha_next : float
        Updated effective miscoverage level α_{t+1}.
    """
    alpha_raw = alpha_t + eta * (alpha_target - ell_t)
    return float(np.clip(alpha_raw, alpha_min, alpha_max))


def optimal_temperature(eta: float, T: int, M: float = 1.0) -> float:
    """Optimal temperature τ* ≍ (η + T^{-1/2}) / M (Remark 7 in the paper).

    Parameters
    ----------
    eta : float
        Step size.
    T : int
        Horizon length.
    M : float
        Bound on the conditional score density ‖f_t‖_∞.

    Returns
    -------
    tau_star : float
    """
    return (eta + T ** (-0.5)) / max(M, 1e-10)
