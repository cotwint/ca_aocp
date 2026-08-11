"""
Tests for ca_aocp.algorithm.CAAOCP -- basic correctness plus the O(K)
complexity fix (bounded score buffer, bounded pi window).
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ca_aocp.algorithm import CAAOCP


def make_series(seed=0, n_pre=200, n_post=200, mean_shift=5.0):
    rng = np.random.default_rng(seed)
    pre = rng.normal(0.0, 1.0, n_pre)
    post = rng.normal(mean_shift, 1.0, n_post)
    return np.concatenate([pre, post])


def test_run_produces_valid_step_results():
    ys = make_series(seed=0)
    model = CAAOCP(alpha=0.1, eta=0.02, decay=0.01, hazard=0.01,
                    max_run_length=30, init_radius=1.0)
    results = model.run(ys)

    assert len(results.steps) == len(ys)
    cov = results.coverage
    assert set(np.unique(cov)).issubset({0, 1})
    assert np.all(results.radii > 0)
    alpha_seq = results.alpha_sequence
    assert np.all(alpha_seq >= model.alpha_min - 1e-9)
    assert np.all(alpha_seq <= model.alpha_max + 1e-9)
    # Long-run coverage should be roughly in the right ballpark (loose
    # sanity bound, not a tight statistical claim).
    assert 0.5 < results.long_run_coverage() < 1.0


def test_score_buffer_is_bounded_by_max_run_length():
    """The whole point of the O(K) fix: the internal score history must
    never exceed max_run_length entries, regardless of series length."""
    K = 25
    ys = make_series(seed=1, n_pre=300, n_post=300)
    model = CAAOCP(alpha=0.1, eta=0.02, decay=0.01, hazard=0.01,
                    max_run_length=K, init_radius=1.0)
    for y in ys:
        model.predict(None)
        model.update(None, float(y))
        assert len(model._scores) <= K


def test_pi_used_in_weighting_is_bounded():
    K = 20
    ys = make_series(seed=2, n_pre=200, n_post=200)
    model = CAAOCP(alpha=0.1, eta=0.02, decay=0.01, hazard=0.01,
                    max_run_length=K, init_radius=1.0)
    for y in ys:
        model.predict(None)
        model.update(None, float(y))
    assert model._pi is not None
    assert len(model._pi) <= K


def test_runtime_scales_sublinearly_vs_series_length():
    """Rough complexity sanity check: with a fixed K, doubling-then-some
    the series length should NOT multiply runtime by anywhere near the
    same factor if per-step cost is O(K) rather than O(t). This is a soft
    check (timing noise exists) with a generous margin, not a tight
    complexity proof.
    """
    K = 30

    def timed_run(T):
        ys = make_series(seed=3, n_pre=T // 2, n_post=T - T // 2)
        model = CAAOCP(alpha=0.1, eta=0.02, decay=0.01, hazard=0.01,
                        max_run_length=K, init_radius=1.0)
        t0 = time.perf_counter()
        model.run(ys)
        return time.perf_counter() - t0

    # warm up (import/JIT-ish effects, first-call overhead)
    timed_run(200)

    t_small = timed_run(400)
    t_large = timed_run(2400)  # 6x the series length

    ratio = t_large / max(t_small, 1e-9)
    # O(t) scaling would predict ratio ~= 6x (or worse, since the old code
    # also grew the *sort* cost each step). O(K) scaling predicts ratio
    # ~= 6x only from the *outer loop* (still O(T) overall, that's
    # unavoidable) but with a much smaller per-step constant that doesn't
    # itself grow with T. We assert the ratio is not drastically worse
    # than the linear-in-T baseline, which would indicate a per-step cost
    # that itself still grows with t.
    assert ratio < 12.0, (
        f"runtime ratio {ratio:.1f}x for a 6x longer series suggests "
        f"per-step cost is still growing with t, not bounded by K"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
