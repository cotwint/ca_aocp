"""
CA-AOCP: Changepoint-Aware Adaptive Online Conformal Prediction
================================================================
Main algorithm class (Algorithm 1 in the paper).

Usage
-----
>>> from ca_aocp.algorithm import CAAOCP
>>> from ca_aocp.predictors import RollingMeanPredictor
>>> model = CAAOCP(alpha=0.1, eta=0.01, decay=0.01, temperature=0.5)
>>> for x_t, y_t in stream:
...     interval = model.predict(x_t)   # before seeing y_t
...     model.update(x_t, y_t)          # after seeing y_t
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np

from .bocpd import GaussianBOCPD
from .weights import compute_weights, effective_sample_size, pre_change_weight_mass
from .conformal import weighted_quantile, prediction_interval, coverage_indicator
from .adaptive_update import smooth_surrogate, adaptive_update
from .predictors import RollingMeanPredictor


# ──────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────

@dataclass
class StepResult:
    """All quantities produced by a single CA-AOCP step."""
    t: int                          # time index (1-based)
    prediction: float               # ŷ_t = f̂_t(X_t)
    radius: float                   # conformal radius q_t
    lower: float                    # interval lower bound
    upper: float                    # interval upper bound
    score: float                    # conformity score S_t (set after observing Y_t)
    covered: int                    # 1{Y_t ∈ Ĉ_t}  (set after observing Y_t)
    alpha_t: float                  # effective miscoverage level used at step t
    alpha_next: float               # updated α_{t+1}
    ell_t: float                    # smooth surrogate loss ℓ_t
    neff: float                     # effective sample size N_eff(t)
    pi: np.ndarray                  # regime relevance scores π_{t,i}
    weights: np.ndarray             # normalised weights w̃_{t,i}


@dataclass
class CAOCPResults:
    """Collected results over a full run."""
    steps: list[StepResult] = field(default_factory=list)

    # ── convenience arrays (populated lazily) ──
    def _arr(self, attr: str) -> np.ndarray:
        return np.array([getattr(s, attr) for s in self.steps])

    @property
    def coverage(self) -> np.ndarray:
        return self._arr("covered")

    @property
    def radii(self) -> np.ndarray:
        return self._arr("radius")

    @property
    def lower_bounds(self) -> np.ndarray:
        return self._arr("lower")

    @property
    def upper_bounds(self) -> np.ndarray:
        return self._arr("upper")

    @property
    def alpha_sequence(self) -> np.ndarray:
        return self._arr("alpha_t")

    @property
    def scores(self) -> np.ndarray:
        return self._arr("score")

    @property
    def neff_sequence(self) -> np.ndarray:
        return self._arr("neff")

    def rolling_coverage(self, window: int = 50) -> np.ndarray:
        """Empirical coverage over a rolling window of length `window`."""
        cov = self.coverage.astype(float)
        return np.convolve(cov, np.ones(window) / window, mode="valid")

    def long_run_coverage(self) -> float:
        """Long-run empirical coverage fraction."""
        return float(self.coverage.mean())


# ──────────────────────────────────────────────────────────────────────
# Main algorithm
# ──────────────────────────────────────────────────────────────────────

class CAAOCP:
    """Changepoint-Aware Adaptive Online Conformal Prediction.

    Implements Algorithm 1 from the paper, combining:
    • Gaussian BOCPD for regime relevance scores π_{t,i}
    • Importance-weighted conformal quantile q_t
    • Smooth ACI update for α_t

    Parameters
    ----------
    alpha : float
        Nominal miscoverage level α ∈ (0, 1).
    eta : float
        ACI step size η > 0.
    decay : float
        Forgetting parameter λ ≥ 0.  λ=0 → pure BOCPD weights.
    temperature : float
        Smooth surrogate temperature τ > 0.
    alpha_min, alpha_max : float | None
        Feasibility bounds.  Default: [α/2, (1+α)/2].
    hazard : float
        BOCPD prior hazard h ∈ (0, 1).
    bocpd_mu0, bocpd_kappa0, bocpd_alpha0, bocpd_beta0 : float
        NIG prior hyperparameters for BOCPD.
    max_run_length : int
        BOCPD truncation depth K.
    predictor : optional
        Base predictor object with .predict() and .update() methods.
        Defaults to a RollingMeanPredictor(window=20).
    init_radius : float
        Initial conformal radius q_1 (warm start).
    true_changepoint : int | None
        If provided, pre-change weight mass is tracked in results
        (for diagnostic purposes only; not used by the algorithm).
    """

    def __init__(
        self,
        alpha: float = 0.1,
        eta: float = 0.02,
        decay: float = 0.01,
        temperature: float = 0.5,
        alpha_min: Optional[float] = None,
        alpha_max: Optional[float] = None,
        hazard: float = 0.01,
        bocpd_mu0: float = 0.0,
        bocpd_kappa0: float = 1.0,
        bocpd_alpha0: float = 1.0,
        bocpd_beta0: float = 1.0,
        max_run_length: int = 500,
        predictor: Any = None,
        init_radius: float = 0.0,
        true_changepoint: Optional[int] = None,
    ) -> None:
        self.alpha = alpha
        self.eta = eta
        self.decay = decay
        self.temperature = temperature
        self.alpha_min = alpha_min if alpha_min is not None else alpha / 2.0
        self.alpha_max = alpha_max if alpha_max is not None else (1.0 + alpha) / 2.0
        self.true_changepoint = true_changepoint

        # Internal state
        self._alpha_t: float = alpha
        self._t: int = 0
        self._scores: list[float] = []
        self._pi: Optional[np.ndarray] = None

        # Components
        self._bocpd = GaussianBOCPD(
            hazard=hazard,
            mu0=bocpd_mu0,
            kappa0=bocpd_kappa0,
            alpha0=bocpd_alpha0,
            beta0=bocpd_beta0,
            max_run_length=max_run_length,
        )
        self._predictor = predictor or RollingMeanPredictor(window=20)
        self._q_t: float = init_radius

    # ------------------------------------------------------------------
    # Public interface (two-call API: predict then update)
    # ------------------------------------------------------------------

    def predict(self, x_t: float | None = None) -> tuple[float, float]:
        """Form the prediction interval for covariate x_t.

        Call this *before* observing Y_t.

        Returns
        -------
        lower, upper : float
            Prediction interval endpoints [ŷ_t − q_t, ŷ_t + q_t].
        """
        yhat = self._predictor.predict(x_t)
        return prediction_interval(yhat, self._q_t)

    def update(self, x_t: float | None, y_t: float) -> StepResult:
        """Process observation (x_t, y_t) and update all internal state.

        Call this *after* observing Y_t.

        Parameters
        ----------
        x_t : float or None
            Covariate (may be None for univariate time series).
        y_t : float
            Observed label.

        Returns
        -------
        result : StepResult
            All quantities from this step.
        """
        self._t += 1
        t = self._t

        # ── Prediction at this step ──
        yhat = self._predictor.predict(x_t)
        lower, upper = prediction_interval(yhat, self._q_t)
        covered = coverage_indicator(y_t, lower, upper)

        # ── Conformity score ──
        s_t = abs(y_t - yhat)
        self._scores.append(s_t)

        # ── Smooth surrogate and alpha update ──
        ell_t = smooth_surrogate(s_t, self._q_t, self.temperature)
        alpha_next = adaptive_update(
            self._alpha_t, ell_t, self.alpha,
            self.eta, self.alpha_min, self.alpha_max,
        )

        # ── Snapshot of current alpha and pi before update ──
        alpha_t_used = self._alpha_t
        pi_used = self._pi.copy() if self._pi is not None else np.array([1.0])
        w_used = (
            compute_weights(pi_used, self.decay, t)
            if len(self._scores) > 1
            else np.array([1.0])
        )
        neff = effective_sample_size(w_used)

        # ── BOCPD update (using S_t just observed) ──
        pi_new = self._bocpd.update(s_t)  # π_{t+1, i} for next step
        self._pi = pi_new

        # ── Compute q_{t+1} for the next step ──
        if len(self._scores) >= 2:
            scores_arr = np.array(self._scores)
            w_next = compute_weights(pi_new, self.decay, t + 1)
            self._q_t = weighted_quantile(scores_arr, w_next, alpha_next)
        else:
            # Only one observation: keep init_radius
            pass

        # ── Update alpha and predictor ──
        self._alpha_t = alpha_next
        self._predictor.update(x_t, y_t)

        return StepResult(
            t=t,
            prediction=yhat,
            radius=self._q_t,
            lower=lower,
            upper=upper,
            score=s_t,
            covered=covered,
            alpha_t=alpha_t_used,
            alpha_next=alpha_next,
            ell_t=ell_t,
            neff=neff,
            pi=pi_used,
            weights=w_used,
        )

    # ------------------------------------------------------------------
    # Batch run helper
    # ------------------------------------------------------------------

    def run(
        self,
        ys: np.ndarray,
        xs: Optional[np.ndarray] = None,
    ) -> CAOCPResults:
        """Run CA-AOCP over a full sequence.

        Parameters
        ----------
        ys : np.ndarray, shape (T,)
            Observed labels Y_1, …, Y_T.
        xs : np.ndarray, shape (T,) | None
            Covariates X_1, …, X_T.  Pass None for univariate series.

        Returns
        -------
        results : CAOCPResults
        """
        results = CAOCPResults()
        T = len(ys)
        for i in range(T):
            x_t = xs[i] if xs is not None else None
            y_t = float(ys[i])
            self.predict(x_t)          # side-effect free at this point
            step = self.update(x_t, y_t)
            results.steps.append(step)
        return results

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def alpha_t(self) -> float:
        """Current effective miscoverage level."""
        return self._alpha_t

    @property
    def q_t(self) -> float:
        """Current conformal radius."""
        return self._q_t

    @property
    def run_length_posterior(self) -> np.ndarray:
        """Current BOCPD run-length posterior P(r_t = k | S_{1:t})."""
        return self._bocpd.get_run_length_posterior()
