"""
Base Predictors
===============
Lightweight online predictors for use with CA-AOCP.

Each predictor implements:
    predict(x_t) → ŷ_t
    update(x_t, y_t) → None

The conformity score is always S_t = |y_t − ŷ_t|.
For plug-in with CA-AOCP, any object with these two methods works.
"""

from __future__ import annotations
from collections import deque
import numpy as np


class RollingMeanPredictor:
    """Predict the rolling mean of the last `window` observations.

    Parameters
    ----------
    window : int
        Look-back window for the rolling mean.
    init_value : float
        Prediction used before `window` observations are available.
    """

    def __init__(self, window: int = 20, init_value: float = 0.0) -> None:
        self.window = window
        self.init_value = init_value
        self._buffer: deque[float] = deque(maxlen=window)

    def predict(self, x_t: float | None = None) -> float:
        """Return the current rolling mean (x_t is ignored; kept for API uniformity)."""
        if len(self._buffer) == 0:
            return self.init_value
        return float(np.mean(self._buffer))

    def update(self, x_t: float | None, y_t: float) -> None:
        """Add y_t to the buffer."""
        self._buffer.append(y_t)


class ExponentialSmoothingPredictor:
    """One-step-ahead prediction via exponential smoothing.

        ŷ_{t+1} = α_s · y_t + (1 − α_s) · ŷ_t

    Parameters
    ----------
    alpha_smooth : float
        Smoothing factor ∈ (0, 1].  α_smooth=1 gives ŷ_{t+1}=y_t.
    init_value : float
        Initial prediction ŷ_1.
    """

    def __init__(self, alpha_smooth: float = 0.3, init_value: float = 0.0) -> None:
        assert 0.0 < alpha_smooth <= 1.0
        self.alpha_smooth = alpha_smooth
        self._yhat: float = init_value

    def predict(self, x_t: float | None = None) -> float:
        return self._yhat

    def update(self, x_t: float | None, y_t: float) -> None:
        self._yhat = self.alpha_smooth * y_t + (1.0 - self.alpha_smooth) * self._yhat


class ZeroPredictor:
    """Always predict zero (treats raw values as conformity scores)."""

    def predict(self, x_t: float | None = None) -> float:
        return 0.0

    def update(self, x_t: float | None, y_t: float) -> None:
        pass


class GlobalMeanPredictor:
    """Predict the global running mean of all observed y values."""

    def __init__(self) -> None:
        self._sum = 0.0
        self._count = 0

    def predict(self, x_t: float | None = None) -> float:
        return self._sum / self._count if self._count > 0 else 0.0

    def update(self, x_t: float | None, y_t: float) -> None:
        self._sum += y_t
        self._count += 1
