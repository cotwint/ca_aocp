"""
Weighted Conformal Calibration
================================
Implements equations (3.5)–(3.7) from the paper.

Given normalised weights w̃_{t,i} and past conformity scores S_{1:t-1},
we form the weighted empirical CDF F̂^w_t and take the
(1 − α_t)-quantile as the conformal radius q_t.

Complexity note (2026-08 follow-up)
------------------------------------
Algorithm 2 in the paper states the weighted quantile can be computed in
"O(K log K) time if the active scores are sorted, or O(K) time with a
linear-time weighted selection routine". The original implementation here
(`method="sort"`, still the default) is the first option: sort-then-cumsum,
O(n log n) in the size of its input. Once callers pass it a K-bounded
buffer (as `ca_aocp.algorithm.CAAOCP` now does), that's O(K log K), which
satisfies the paper's stated complexity claim -- but only the looser of
its two explicitly-allowed options, not the tighter O(K) one. `method="select"`
adds that second option: an expected-linear-time weighted selection
(quickselect-style recursive partitioning around a random pivot, in the
spirit of weighted-median-of-medians algorithms), so the O(K) claim can be
made without qualification if that matters. It is *not* the default,
because for the small K this project actually uses (tens to a few hundred),
NumPy's C-level `argsort` on `method="sort"` is faster in wall-clock terms
than a Python-level quickselect loop despite the worse asymptotic exponent
-- the two methods are provided so the choice between "asymptotically
tighter" and "faster in practice at realistic K" can be made explicitly by
the caller, and are cross-checked for agreement in `tests/test_weights_conformal.py`.
"""

from __future__ import annotations
from typing import Optional
import numpy as np


def weighted_quantile(
    scores: np.ndarray,
    weights: np.ndarray,
    level: float,
    method: str = "sort",
    rng: Optional[np.random.Generator] = None,
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
    method : {"sort", "select"}
        "sort" (default): sort-then-cumsum, O(n log n) -- matches the
        "O(K log K) if the active scores are sorted" branch of Algorithm 2.
        "select": expected O(n) weighted quickselect -- matches the
        "O(K) with a linear-time weighted selection routine" branch.
        Both implement the exact same definition and agree to floating-point
        tolerance (see tests); "select" exists to let callers claim the
        tighter complexity bound when that matters more than practical
        constant-factor speed at small n.
    rng : np.random.Generator, optional
        Only used by `method="select"`, for random pivot choice. Defaults
        to a fresh `np.random.default_rng()` if not supplied.

    Returns
    -------
    q : float
        Conformal radius q_t.
    """
    assert len(scores) == len(weights), "scores and weights must have the same length."
    assert 0.0 < level < 1.0, "level must be in (0, 1)."

    if method == "select":
        return _weighted_quantile_select(scores, weights, level, rng)
    if method != "sort":
        raise ValueError(f"unknown method {method!r}; expected 'sort' or 'select'.")

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


def _weighted_quantile_select(
    scores: np.ndarray,
    weights: np.ndarray,
    level: float,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Expected-O(n) weighted quantile via quickselect-style partitioning.

    Finds the smallest value v among `scores` such that the total weight of
    entries with score <= v is >= (1 - level) -- the same target as the
    sort-based implementation, computed without a full sort. At each step,
    partition the remaining candidates around a randomly chosen pivot value
    into (< pivot, == pivot, > pivot); the target weight tells us which
    partition must contain the answer, so we recurse into only that one
    partition (discarding the other two), same as unweighted quickselect
    but tracking cumulative weight instead of cumulative count. Expected
    O(n) total work because the candidate set shrinks by a constant factor
    in expectation each round (as in standard randomized quickselect);
    worst case O(n^2) for adversarial pivot sequences, which random pivot
    selection makes exponentially unlikely.
    """
    if rng is None:
        rng = np.random.default_rng()

    cand_vals = np.asarray(scores, dtype=float)
    cand_wts = np.asarray(weights, dtype=float)
    target = 1.0 - level
    tol = 1e-12

    while True:
        n = len(cand_vals)
        if n == 1:
            return float(cand_vals[0])

        pivot_val = cand_vals[rng.integers(0, n)]
        less_mask = cand_vals < pivot_val
        equal_mask = cand_vals == pivot_val
        greater_mask = cand_vals > pivot_val

        w_less = float(cand_wts[less_mask].sum())
        w_equal = float(cand_wts[equal_mask].sum())

        if target <= w_less + tol:
            if not less_mask.any():
                # Degenerate/numerical-noise case: target is ~0 but no
                # element is strictly smaller than the pivot -- the answer
                # is the smallest candidate value.
                return float(cand_vals.min())
            cand_vals = cand_vals[less_mask]
            cand_wts = cand_wts[less_mask]
            continue

        if target <= w_less + w_equal + tol:
            return float(pivot_val)

        # Answer lies strictly beyond the pivot; recurse right with the
        # remaining target weight.
        target -= w_less + w_equal
        if not greater_mask.any():
            # Numerical edge case (e.g. level very close to 0, weights not
            # summing to exactly 1 due to floating-point error): fall back
            # to the largest candidate, mirroring the sort-based
            # implementation's `idx = min(idx, len - 1)` clamp.
            return float(cand_vals.max())
        cand_vals = cand_vals[greater_mask]
        cand_wts = cand_wts[greater_mask]


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
