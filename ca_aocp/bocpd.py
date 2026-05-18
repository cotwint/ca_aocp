"""
Bayesian Online Changepoint Detection (BOCPD)
=============================================
Implements the Adams & MacKay (2007) algorithm with a
Normal-Inverse-Gamma (NIG) conjugate prior, yielding a
Student-t predictive distribution.

Reference
---------
Adams, R. P. & MacKay, D. J. C. (2007).
Bayesian online changepoint detection. arXiv:0710.3742.
"""

from __future__ import annotations
import numpy as np
from scipy import stats
from typing import Optional


class NIGParams:
    """Normal-Inverse-Gamma sufficient statistics for one run.

    Parameters
    ----------
    mu0, kappa0, alpha0, beta0 : float
        Prior hyperparameters.  The predictive distribution is
        Student-t(2*alpha, mu, beta*(kappa+1)/(alpha*kappa)).
    """

    __slots__ = ("mu", "kappa", "alpha", "beta", "n")

    def __init__(
        self,
        mu0: float = 0.0,
        kappa0: float = 1.0,
        alpha0: float = 1.0,
        beta0: float = 1.0,
    ) -> None:
        self.mu = mu0
        self.kappa = kappa0
        self.alpha = alpha0
        self.beta = beta0
        self.n = 0  # observations in this run

    def update(self, x: float) -> "NIGParams":
        """Return a *new* NIGParams after observing x (non-destructive)."""
        p = NIGParams.__new__(NIGParams)
        kappa_new = self.kappa + 1.0
        p.mu = (self.kappa * self.mu + x) / kappa_new
        p.kappa = kappa_new
        p.alpha = self.alpha + 0.5
        p.beta = self.beta + (self.kappa * (x - self.mu) ** 2) / (2.0 * kappa_new)
        p.n = self.n + 1
        return p

    def log_pred(self, x: float) -> float:
        """Log predictive density p(x | NIG params) = Student-t."""
        # Student-t(df=2*alpha, loc=mu, scale=sqrt(beta*(kappa+1)/(alpha*kappa)))
        df = 2.0 * self.alpha
        scale = np.sqrt(self.beta * (self.kappa + 1.0) / (self.alpha * self.kappa))
        return stats.t.logpdf(x, df=df, loc=self.mu, scale=scale)


class GaussianBOCPD:
    """Online BOCPD with a NIG conjugate prior on Gaussian observations.

    The run-length posterior is maintained as a log-probability vector
    and truncated to ``max_run_length`` entries for O(K) per-step cost.

    Parameters
    ----------
    hazard : float
        Prior probability of a changepoint at each step (h in the paper).
    mu0, kappa0, alpha0, beta0 : float
        NIG prior hyperparameters shared across all runs.
    max_run_length : int
        Truncation depth K.  Run lengths > K are discarded.
    """

    def __init__(
        self,
        hazard: float = 0.01,
        mu0: float = 0.0,
        kappa0: float = 1.0,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        max_run_length: int = 500,
    ) -> None:
        self.h = hazard
        self.K = max_run_length
        self._prior = NIGParams(mu0, kappa0, alpha0, beta0)

        # log P(r_{t-1} = k | S_{1:t-1}) as a 1-D array indexed by k.
        # At t=1 (before any observation), r_0 = 0 with probability 1.
        self._log_R: np.ndarray = np.array([0.0])  # shape (1,)

        # NIG params for each active run length; index k <-> run length k.
        self._nig: list[NIGParams] = [NIGParams(
            self._prior.mu, self._prior.kappa,
            self._prior.alpha, self._prior.beta,
        )]

        self.t: int = 0  # steps processed so far

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, x: float) -> np.ndarray:
        """Process one new observation x = S_t.

        Returns
        -------
        pi : np.ndarray, shape (t,)
            Regime relevance scores π_{t+1, i} for i = 1, …, t, where
            π_{t+1, i} = P(r_t ≥ t + 1 − i | S_{1:t}).
            Index 0 corresponds to the most recent observation (i = t).
        """
        # ── Step 1: predictive log-likelihoods for each active run ──
        log_preds = np.array([nig.log_pred(x) for nig in self._nig])

        # Align _log_R to the length of active NIGs (truncation may leave
        # _log_R longer with -inf padding; extract only active entries).
        log_R_active = self._log_R[:len(self._nig)]

        # ── Step 2: growth probabilities (run length increases by 1) ──
        log_growth = log_R_active + np.log(1.0 - self.h) + log_preds

        # ── Step 3: changepoint probability (run resets to 0) ──
        log_cp = np.logaddexp.reduce(log_R_active) + np.log(self.h) + self._prior.log_pred(x)

        # ── Step 4: concatenate [cp, growth] and normalise ──
        log_R_new = np.empty(len(log_growth) + 1)
        log_R_new[0] = log_cp
        log_R_new[1:] = log_growth
        log_R_new -= np.logaddexp.reduce(log_R_new)  # normalise in log-space

        # ── Step 5: update NIG sufficient statistics ──
        nig_new: list[NIGParams] = [
            NIGParams(self._prior.mu, self._prior.kappa,
                      self._prior.alpha, self._prior.beta)  # new run (k=0)
        ]
        nig_new.extend(nig.update(x) for nig in self._nig)

        # ── Step 6: truncate to K most-probable run lengths ──
        if len(log_R_new) > self.K:
            keep = np.argpartition(log_R_new, -self.K)[-self.K:]
            keep = np.sort(keep)
            log_R_new_trunc = log_R_new[keep]
            log_R_new_trunc -= np.logaddexp.reduce(log_R_new_trunc)
            nig_new = [nig_new[k] for k in keep]
            # Rebuild dense array; entries not in keep get -inf
            log_R_dense = np.full(len(log_R_new), -np.inf)
            log_R_dense[keep] = log_R_new_trunc
            log_R_new = log_R_dense

        self._log_R = log_R_new
        self._nig = nig_new
        self.t += 1

        return self._compute_pi()

    def get_run_length_posterior(self) -> np.ndarray:
        """Return P(r_{t-1} = k | S_{1:t-1}) as a probability array."""
        return np.exp(self._log_R)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_pi(self) -> np.ndarray:
        """Compute π_{t+1, i} = P(r_t ≥ t+1−i | S_{1:t}) for i=1,…,t.

        Returns
        -------
        pi : np.ndarray, shape (t,)
            pi[j] corresponds to observation index i = t - j,
            i.e. pi[0] is for the most recent past observation.
        """
        R = np.exp(self._log_R)  # shape (t+1,) at most
        # R[k] = P(r_t = k | ...).  We want the survival function:
        # π_{t+1, i} = P(r_t ≥ t+1−i) = sum_{k ≥ t+1−i} R[k]
        # Equivalently, using reverse cumulative sum:
        # For observation at position i (1-indexed from the start),
        # the minimum run length needed is t+1-i.
        # We build a dense version of R padded to length t+1.
        t = self.t
        R_dense = np.zeros(t + 1)
        R_dense[: len(R)] = R

        # Survival function: sf[k] = P(r_t >= k) = sum_{j>=k} R[j]
        # sf is the reverse cumulative sum (suffix sum).
        sf = np.cumsum(R_dense[::-1])[::-1]  # shape (t+1,)

        # π_{t+1, i} = sf[t+1-i].  For i=1,...,t this is sf[t], sf[t-1], ..., sf[1].
        # We return pi as an array of length t indexed by [i-1] for i=1,...,t.
        pi = sf[1: t + 1][::-1]  # pi[i-1] = π_{t+1, i}
        return np.clip(pi, 0.0, 1.0)
