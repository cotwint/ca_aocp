"""Basic sanity tests for ca_aocp.weights / ca_aocp.conformal, unaffected
by the BOCPD fix but worth pinning down given they're on the critical
path for the O(K) change (compute_weights/weighted_quantile now typically
receive bounded-length inputs from CAAOCP instead of full-history ones)."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ca_aocp.weights import compute_weights, effective_sample_size
from ca_aocp.conformal import weighted_quantile, prediction_interval, coverage_indicator


def test_compute_weights_sums_to_one_and_handles_short_windows():
    pi = np.array([0.2, 0.5, 0.9, 1.0])  # e.g. from pi_recent(window=4)
    w = compute_weights(pi, decay=0.01, t=100)
    assert len(w) == len(pi)
    assert abs(w.sum() - 1.0) < 1e-9
    assert np.all(w >= 0)


def test_compute_weights_prefers_recent_with_decay():
    pi = np.ones(10)  # uniform BOCPD relevance
    w = compute_weights(pi, decay=0.2, t=50)
    # With decay>0 and uniform pi, weight should be monotonically
    # increasing toward the most recent (last) observation.
    assert np.all(np.diff(w) >= -1e-12)


def test_weighted_quantile_matches_unweighted_for_uniform_weights():
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = np.ones(5) / 5
    q = weighted_quantile(scores, weights, level=0.4)  # 60th percentile
    # cdf after sorting: [.2,.4,.6,.8,1.0]; target=0.6 -> idx=2 -> score=3.0
    assert q == pytest.approx(3.0)


def test_prediction_interval_and_coverage():
    lo, hi = prediction_interval(prediction=10.0, radius=2.0)
    assert (lo, hi) == (8.0, 12.0)
    assert coverage_indicator(9.0, lo, hi) == 1
    assert coverage_indicator(15.0, lo, hi) == 0


# ── weighted_quantile: method="select" (2026-08 follow-up, O(K) complexity fix) ──
# Algorithm 2 in the paper allows either "O(K log K) if sorted" (method="sort",
# the pre-existing default) or "O(K) with a linear-time weighted selection
# routine" (method="select", added below). These tests check the two agree,
# since "select" is a from-scratch reimplementation of the same definition
# rather than a refactor of the sort-based one.

def test_select_matches_sort_on_basic_case():
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = np.ones(5) / 5
    q_sort = weighted_quantile(scores, weights, level=0.4, method="sort")
    q_select = weighted_quantile(scores, weights, level=0.4, method="select",
                                  rng=np.random.default_rng(0))
    assert q_select == pytest.approx(q_sort)


def test_select_matches_sort_randomized():
    """Cross-check method='select' against method='sort' over many random
    (scores, weights, level) triples, including duplicate score values
    (ties) which are the trickiest case for the partition-based selection
    logic to get exactly right."""
    master_rng = np.random.default_rng(42)
    for trial in range(200):
        n = master_rng.integers(1, 40)
        # Bias toward small integer-valued scores so duplicates are common.
        scores = master_rng.integers(0, 6, size=n).astype(float)
        raw_w = master_rng.exponential(1.0, size=n)
        weights = raw_w / raw_w.sum()
        level = float(master_rng.uniform(0.01, 0.99))

        q_sort = weighted_quantile(scores, weights, level=level, method="sort")
        q_select = weighted_quantile(
            scores, weights, level=level, method="select",
            rng=np.random.default_rng(trial),
        )
        assert q_select == pytest.approx(q_sort), (
            f"trial {trial}: sort={q_sort} select={q_select} "
            f"scores={scores.tolist()} weights={weights.tolist()} level={level}"
        )


def test_select_matches_sort_at_extreme_levels():
    """level close to 0 (target close to 1, near the max score) and level
    close to 1 (target close to 0, near the min score) exercise the
    fallback/clamp branches in both implementations."""
    rng = np.random.default_rng(7)
    scores = rng.normal(0, 1, 25)
    raw_w = rng.exponential(1.0, 25)
    weights = raw_w / raw_w.sum()

    for level in (0.001, 0.01, 0.99, 0.999):
        q_sort = weighted_quantile(scores, weights, level=level, method="sort")
        q_select = weighted_quantile(scores, weights, level=level, method="select",
                                      rng=np.random.default_rng(1))
        assert q_select == pytest.approx(q_sort), f"level={level}"


def test_select_single_element():
    q = weighted_quantile(np.array([3.5]), np.array([1.0]), level=0.1, method="select")
    assert q == pytest.approx(3.5)


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        weighted_quantile(np.array([1.0, 2.0]), np.array([0.5, 0.5]), level=0.1,
                           method="bogus")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
