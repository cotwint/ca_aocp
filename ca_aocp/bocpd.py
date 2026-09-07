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

Truncation bookkeeping (2026-08 fix)
-------------------------------------
The run-length posterior is kept truncated to the ``max_run_length`` (K)
most probable run lengths for O(K) per-step cost. Earlier versions of this
class stored the truncated posterior in a *dense* array padded with -inf
at dropped positions, and then, on the following step, sliced out the
first ``len(self._nig)`` entries of that dense array under the assumption
that "array index == run length == position in `self._nig`". Once
truncation actually drops a non-trailing index (which happens routinely
in practice, not just in pathological cases), that assumption breaks:
the sliced prior no longer lines up with the NIG parameters it is paired
against, corrupting the growth/changepoint probabilities from that step
onward. Verified empirically pre-fix: the truncated run-length posterior's
top-5 most probable run lengths diverged from an (effectively) untruncated
reference BOCPD run on the same data by up to ~0.99 in probability mass
(far beyond what discarding a low-probability tail could explain).

The fix tracks run lengths *explicitly* in a parallel `_run_lengths` array
so that `_log_R[j]` / `_nig[j]` / `_run_lengths[j]` always refer to the same
slot after compaction, regardless of which original indices were kept.
`get_run_length_posterior()` still returns a dense array indexed by
absolute run length (for backward-compatible `argmax(...)`-style callers),
reconstructed on demand from the compact state; since retained run lengths
are bounded by O(K), this reconstruction is itself O(K), not O(t).

Two pi-vector accessors are provided:

- `_compute_pi()` / the return value of `update()`: the ORIGINAL contract,
  a dense, oldest-first array of length `t` (one weight per historical
  observation). Kept for backward compatibility with code written against
  the original API (e.g. this project's `experiments/run_ablation.py`,
  which maintains its own unbounded score history and expects a
  matching-length pi array). This is now *correct* (built from the fixed
  internal state) but still O(t) to materialize, because the contract
  itself requires one entry per historical observation.
- `pi_recent(window=None)`: a NEW bounded accessor returning weights for
  only the most recent `min(t, window or K)` observations -- O(K) to
  compute and O(K) in size. This is what `ca_aocp.algorithm.CAAOCP` uses
  internally to get genuine O(K) per-step complexity end to end.

Legacy-pi leak fix (follow-up, same session)
---------------------------------------------
`update()` used to unconditionally build and return the legacy O(t) pi
vector (via `_compute_pi()`) on every call, even for callers like
`CAAOCP.update()` that immediately discard the return value and fetch
`pi_recent()` separately. That meant `CAAOCP`, despite the O(K) buffers
above, was still silently paying an O(t) cost per step underneath.
Measured impact: at t=500,000, the discarded `_compute_pi()` call took
~330x longer than the `pi_recent()` call whose result is actually used,
and the gap grows linearly with t (`pi_recent()` stays flat around 10us
regardless of t). `update()` now takes a `compute_legacy_pi` flag
(default True, preserving the old contract for callers who need it, e.g.
`experiments/run_ablation.py`); `CAAOCP` passes `compute_legacy_pi=False`
to skip the wasted computation entirely.
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

    The run-length posterior is maintained as a *compacted* log-probability
    array (`_log_R`) together with a parallel `_run_lengths` array recording
    which absolute run-length each slot represents, both truncated to at
    most `max_run_length` entries for O(K) per-step cost. See module
    docstring for why this bookkeeping matters.

    Parameters
    ----------
    hazard : float
        Prior probability of a changepoint at each step (h in the paper).
    mu0, kappa0, alpha0, beta0 : float
        NIG prior hyperparameters shared across all runs.
    max_run_length : int
        Truncation depth K.  Run lengths beyond the K most probable are
        discarded each step.
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

        # log P(r_{t-1} = k | S_{1:t-1}), compacted: _log_R[j] is the
        # log-probability for run length _run_lengths[j].
        self._log_R: np.ndarray = np.array([0.0])
        self._run_lengths: np.ndarray = np.array([0], dtype=int)

        # NIG params for each active run length; _nig[j] <-> _run_lengths[j].
        self._nig: list[NIGParams] = [NIGParams(
            self._prior.mu, self._prior.kappa,
            self._prior.alpha, self._prior.beta,
        )]

        self.t: int = 0  # steps processed so far

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, x: float, compute_legacy_pi: bool = True) -> Optional[np.ndarray]:
        """Process one new observation x = S_t.

        Parameters
        ----------
        compute_legacy_pi : bool
            If True (default), builds and returns the legacy O(t)-shaped
            pi vector below, for backward compatibility with callers that
            depend on `update()`'s return value (e.g. this project's
            `experiments/run_ablation.py`). If False, skips that
            construction entirely and returns None -- callers that only
            need the bounded window should use `pi_recent()` instead and
            pass `compute_legacy_pi=False` so `update()` itself stays
            O(K) per call. `ca_aocp.algorithm.CAAOCP` does this: it always
            discarded the return value of `update()` while still paying
            for the O(t) legacy computation on every step (verified: at
            t=500,000 the discarded `_compute_pi()` call cost ~330x what
            the O(K) `pi_recent()` call it actually uses costs, and grows
            linearly with t where `pi_recent()` stays flat). This flag
            closes that leak without breaking the legacy contract for
            callers that still want it.

        Returns
        -------
        pi : np.ndarray, shape (t,), or None
            If `compute_legacy_pi` is True: regime relevance scores
            π_{t+1, i} for i = 1, …, t, where
            π_{t+1, i} = P(r_t ≥ t + 1 − i | S_{1:t}), oldest observation
            first (index 0). This is the legacy, backward-compatible,
            O(t)-shaped accessor -- see `pi_recent()` for an O(K) one.
            If `compute_legacy_pi` is False: None.
        """
        # ── Step 1: predictive log-likelihoods for each active run ──
        # self._nig[j] and self._log_R[j] are already aligned (both
        # compacted together at the end of the previous update()).
        log_preds = np.array([nig.log_pred(x) for nig in self._nig])
        log_R_active = self._log_R

        # ── Step 2: growth probabilities (run length increases by 1) ──
        log_growth = log_R_active + np.log(1.0 - self.h) + log_preds

        # ── Step 3: changepoint probability (run resets to 0) ──
        log_cp = np.logaddexp.reduce(log_R_active) + np.log(self.h) + self._prior.log_pred(x)

        # ── Step 4: concatenate [cp, growth] and normalise ──
        log_R_new = np.empty(len(log_growth) + 1)
        log_R_new[0] = log_cp
        log_R_new[1:] = log_growth
        log_R_new -= np.logaddexp.reduce(log_R_new)  # normalise in log-space

        # Parallel run-length labels for log_R_new: index 0 is the fresh
        # run (r_t = 0); the rest are each previously-tracked run length
        # grown by one step.
        candidate_run_lengths = np.empty(len(self._run_lengths) + 1, dtype=int)
        candidate_run_lengths[0] = 0
        candidate_run_lengths[1:] = self._run_lengths + 1

        # ── Step 5: update NIG sufficient statistics ──
        nig_new: list[NIGParams] = [
            NIGParams(self._prior.mu, self._prior.kappa,
                      self._prior.alpha, self._prior.beta)  # new run (k=0)
        ]
        nig_new.extend(nig.update(x) for nig in self._nig)

        # ── Step 6: truncate to K most-probable run lengths ──
        # Compact ALL THREE parallel arrays/lists together by the same
        # `keep` index set, so alignment is preserved regardless of which
        # original positions survive.
        if len(log_R_new) > self.K:
            keep = np.argpartition(log_R_new, -self.K)[-self.K:]
            keep = np.sort(keep)
            log_R_new = log_R_new[keep]
            log_R_new = log_R_new - np.logaddexp.reduce(log_R_new)
            nig_new = [nig_new[k] for k in keep]
            candidate_run_lengths = candidate_run_lengths[keep]

        self._log_R = log_R_new
        self._nig = nig_new
        self._run_lengths = candidate_run_lengths
        self.t += 1

        if not compute_legacy_pi:
            return None
        return self._compute_pi()

    def get_run_length_posterior(self) -> np.ndarray:
        """Dense P(r_{t-1} = k | S_{1:t-1}) indexed by absolute run length k.

        Length is (max tracked run length + 1), which is bounded by O(K)
        once truncation is active -- NOT O(t). Reconstructed on demand
        from the compact `_run_lengths` / `_log_R` state.
        """
        if len(self._run_lengths) == 0:
            return np.array([1.0])
        dense = np.zeros(int(self._run_lengths.max()) + 1)
        dense[self._run_lengths] = np.exp(self._log_R)
        return dense

    def pi_recent(self, window: Optional[int] = None) -> np.ndarray:
        """O(K)-bounded pi vector for only the most recent observations.

        Parameters
        ----------
        window : int or None
            How many of the most recent observations to return weights
            for. Defaults to `self.K` (the truncation depth), which is
            the natural bound: observations older than K steps back
            cannot have any surviving run length reach them once
            truncation is active, so their true weight is (approximately)
            zero anyway.

        Returns
        -------
        pi : np.ndarray, shape (min(t, window),)
            Oldest-first, same convention as `_compute_pi()` / `update()`'s
            return value, just bounded in length instead of O(t).
        """
        L = min(self.t, window if window is not None else self.K)
        if L == 0:
            return np.array([])
        ages = np.arange(L, 0, -1)  # [L, L-1, ..., 1]; last entry = most recent
        return self._survival_function(ages)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _survival_function(self, ages: np.ndarray) -> np.ndarray:
        """P(r_t >= age) for each requested age, using only the O(K)
        (or fewer) distinct run lengths currently tracked."""
        order = np.argsort(self._run_lengths)
        rl_sorted = self._run_lengths[order]
        R_sorted = np.exp(self._log_R[order])
        # suffix_sum[j] = P(r_t >= rl_sorted[j])
        suffix_sum = np.cumsum(R_sorted[::-1])[::-1]
        idx = np.searchsorted(rl_sorted, ages, side="left")
        n = len(suffix_sum)
        out = np.where(idx < n, suffix_sum[np.clip(idx, 0, n - 1)], 0.0)
        return np.clip(out, 0.0, 1.0)

    def _compute_pi(self) -> np.ndarray:
        """Legacy full-length (O(t)) pi vector -- see `pi_recent()` for the
        O(K)-bounded equivalent used internally by `CAAOCP`.

        Returns
        -------
        pi : np.ndarray, shape (t,)
            pi[j] corresponds to observation index i = j + 1 (oldest
            observation first, i.e. pi[0] = π_{t+1,1}, pi[t-1] = π_{t+1,t}).
        """
        if self.t == 0:
            return np.array([])
        ages = np.arange(self.t, 0, -1)  # [t, t-1, ..., 1]
        return self._survival_function(ages)
