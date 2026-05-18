"""
ca_aocp
=======
Changepoint-Aware Adaptive Online Conformal Prediction.

Modules
-------
bocpd           Bayesian Online Changepoint Detection (BOCPD).
weights         Changepoint-aware importance weight construction.
conformal       Weighted conformal calibration and prediction intervals.
adaptive_update Smooth ACI miscoverage level update.
predictors      Lightweight online base predictors.
algorithm       Main CA-AOCP algorithm class.
"""

from .algorithm import CAAOCP, CAOCPResults, StepResult
from .bocpd import GaussianBOCPD
from .weights import compute_weights, effective_sample_size
from .conformal import weighted_quantile, prediction_interval
from .adaptive_update import smooth_surrogate, adaptive_update, optimal_temperature
from .predictors import (
    RollingMeanPredictor,
    ExponentialSmoothingPredictor,
    ZeroPredictor,
    GlobalMeanPredictor,
)

__all__ = [
    "CAAOCP", "CAOCPResults", "StepResult",
    "GaussianBOCPD",
    "compute_weights", "effective_sample_size",
    "weighted_quantile", "prediction_interval",
    "smooth_surrogate", "adaptive_update", "optimal_temperature",
    "RollingMeanPredictor", "ExponentialSmoothingPredictor",
    "ZeroPredictor", "GlobalMeanPredictor",
]
