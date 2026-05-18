"""
Baseline Methods for Comparison
=================================
ACI          Standard Adaptive Conformal Inference (Gibbs & Candès, 2021).
ACISlidingWindow  ACI with a fixed-length sliding window of uniform weights.
EWMACP       Exponential weighted moving average conformal prediction.
"""

from __future__ import annotations
from typing import Any, Optional
from dataclasses import dataclass, field
import numpy as np

from ca_aocp.conformal import weighted_quantile, prediction_interval, coverage_indicator
from ca_aocp.adaptive_update import smooth_surrogate, adaptive_update
from ca_aocp.predictors import RollingMeanPredictor


# ──────────────────────────────────────────────────────────────────────
# Shared result container
# ──────────────────────────────────────────────────────────────────────

@dataclass
class BaselineResults:
    coverages: list[int] = field(default_factory=list)
    radii: list[float] = field(default_factory=list)
    lowers: list[float] = field(default_factory=list)
    uppers: list[float] = field(default_factory=list)
    alpha_seq: list[float] = field(default_factory=list)

    def rolling_coverage(self, window: int = 50) -> np.ndarray:
        cov = np.array(self.coverages, dtype=float)
        return np.convolve(cov, np.ones(window) / window, mode="valid")

    def long_run_coverage(self) -> float:
        return float(np.mean(self.coverages))


# ──────────────────────────────────────────────────────────────────────
# Standard ACI (uniform weights over all history)
# ──────────────────────────────────────────────────────────────────────

class ACI:
    """Standard Adaptive Conformal Inference (Gibbs & Candès, 2021).

    Uses uniform weights over the full score history.

    Parameters
    ----------
    alpha, eta, temperature, alpha_min, alpha_max : float
        Same semantics as CAAOCP.
    predictor : optional base predictor.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        eta: float = 0.02,
        temperature: float = 0.5,
        alpha_min: Optional[float] = None,
        alpha_max: Optional[float] = None,
        predictor: Any = None,
        init_radius: float = 0.0,
    ) -> None:
        self.alpha = alpha
        self.eta = eta
        self.temperature = temperature
        self.alpha_min = alpha_min if alpha_min is not None else alpha / 2.0
        self.alpha_max = alpha_max if alpha_max is not None else (1.0 + alpha) / 2.0
        self._alpha_t = alpha
        self._q_t = init_radius
        self._scores: list[float] = []
        self._predictor = predictor or RollingMeanPredictor(window=20)

    def run(self, ys: np.ndarray, xs: Optional[np.ndarray] = None) -> BaselineResults:
        results = BaselineResults()
        for i, y_t in enumerate(ys):
            x_t = xs[i] if xs is not None else None
            yhat = self._predictor.predict(x_t)
            lo, hi = prediction_interval(yhat, self._q_t)
            s_t = abs(float(y_t) - yhat)
            covered = coverage_indicator(float(y_t), lo, hi)
            ell_t = smooth_surrogate(s_t, self._q_t, self.temperature)
            alpha_next = adaptive_update(
                self._alpha_t, ell_t, self.alpha,
                self.eta, self.alpha_min, self.alpha_max,
            )
            self._scores.append(s_t)
            if len(self._scores) >= 2:
                w = np.ones(len(self._scores)) / len(self._scores)
                self._q_t = weighted_quantile(
                    np.array(self._scores), w, alpha_next,
                )
            self._alpha_t = alpha_next
            self._predictor.update(x_t, float(y_t))
            results.coverages.append(covered)
            results.radii.append(self._q_t)
            results.lowers.append(lo)
            results.uppers.append(hi)
            results.alpha_seq.append(self._alpha_t)
        return results


# ──────────────────────────────────────────────────────────────────────
# ACI with sliding window
# ──────────────────────────────────────────────────────────────────────

class ACISlidingWindow:
    """ACI with a uniform sliding window of length W.

    Theorem 3(ii) in the paper shows this requires Θ(W) steps to recover
    after a changepoint — the lower-bound complement to CA-AOCP.

    Parameters
    ----------
    window : int
        Window length W.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        eta: float = 0.02,
        temperature: float = 0.5,
        window: int = 100,
        alpha_min: Optional[float] = None,
        alpha_max: Optional[float] = None,
        predictor: Any = None,
        init_radius: float = 0.0,
    ) -> None:
        self.alpha = alpha
        self.eta = eta
        self.temperature = temperature
        self.window = window
        self.alpha_min = alpha_min if alpha_min is not None else alpha / 2.0
        self.alpha_max = alpha_max if alpha_max is not None else (1.0 + alpha) / 2.0
        self._alpha_t = alpha
        self._q_t = init_radius
        self._scores: list[float] = []
        self._predictor = predictor or RollingMeanPredictor(window=20)

    def run(self, ys: np.ndarray, xs: Optional[np.ndarray] = None) -> BaselineResults:
        results = BaselineResults()
        for i, y_t in enumerate(ys):
            x_t = xs[i] if xs is not None else None
            yhat = self._predictor.predict(x_t)
            lo, hi = prediction_interval(yhat, self._q_t)
            s_t = abs(float(y_t) - yhat)
            covered = coverage_indicator(float(y_t), lo, hi)
            ell_t = smooth_surrogate(s_t, self._q_t, self.temperature)
            alpha_next = adaptive_update(
                self._alpha_t, ell_t, self.alpha,
                self.eta, self.alpha_min, self.alpha_max,
            )
            self._scores.append(s_t)
            window_scores = np.array(self._scores[-self.window:])
            w = np.ones(len(window_scores)) / len(window_scores)
            self._q_t = weighted_quantile(window_scores, w, alpha_next)
            self._alpha_t = alpha_next
            self._predictor.update(x_t, float(y_t))
            results.coverages.append(covered)
            results.radii.append(self._q_t)
            results.lowers.append(lo)
            results.uppers.append(hi)
            results.alpha_seq.append(self._alpha_t)
        return results


# ──────────────────────────────────────────────────────────────────────
# EWMA-CP (exponential decay, no changepoint detection)
# ──────────────────────────────────────────────────────────────────────

class EWMACP:
    """EWMA-based conformal prediction (no changepoint detection).

    Weights: w_{t,i} ∝ exp(−λ(t−i)), uniform over regime relevance (π=1).
    This is the Proposition B.2(iii) reduction of CA-AOCP.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        eta: float = 0.02,
        temperature: float = 0.5,
        decay: float = 0.01,
        alpha_min: Optional[float] = None,
        alpha_max: Optional[float] = None,
        predictor: Any = None,
        init_radius: float = 0.0,
    ) -> None:
        self.alpha = alpha
        self.eta = eta
        self.temperature = temperature
        self.decay = decay
        self.alpha_min = alpha_min if alpha_min is not None else alpha / 2.0
        self.alpha_max = alpha_max if alpha_max is not None else (1.0 + alpha) / 2.0
        self._alpha_t = alpha
        self._q_t = init_radius
        self._scores: list[float] = []
        self._predictor = predictor or RollingMeanPredictor(window=20)

    def run(self, ys: np.ndarray, xs: Optional[np.ndarray] = None) -> BaselineResults:
        results = BaselineResults()
        for i, y_t in enumerate(ys):
            x_t = xs[i] if xs is not None else None
            yhat = self._predictor.predict(x_t)
            lo, hi = prediction_interval(yhat, self._q_t)
            s_t = abs(float(y_t) - yhat)
            covered = coverage_indicator(float(y_t), lo, hi)
            ell_t = smooth_surrogate(s_t, self._q_t, self.temperature)
            alpha_next = adaptive_update(
                self._alpha_t, ell_t, self.alpha,
                self.eta, self.alpha_min, self.alpha_max,
            )
            self._scores.append(s_t)
            n = len(self._scores)
            lags = np.arange(n - 1, -1, -1, dtype=float)
            w = np.exp(-self.decay * lags)
            w /= w.sum()
            self._q_t = weighted_quantile(np.array(self._scores), w, alpha_next)
            self._alpha_t = alpha_next
            self._predictor.update(x_t, float(y_t))
            results.coverages.append(covered)
            results.radii.append(self._q_t)
            results.lowers.append(lo)
            results.uppers.append(hi)
            results.alpha_seq.append(self._alpha_t)
        return results
