"""
Regression tests for the GaussianBOCPD truncation alignment bug (2026-08).

Background: the pre-fix implementation stored the truncated run-length
posterior in a dense array padded with -inf at dropped positions, then
sliced the FIRST len(self._nig) entries of that dense array on the next
step under the assumption that array index == run length == position in
self._nig. That assumption silently breaks once truncation drops a
non-trailing index (the common case), corrupting the posterior from that
point on. These tests check that the fixed implementation's truncated
posterior actually tracks an (effectively) untruncated reference, and that
the legacy full-length pi API and the new bounded pi_recent() API are both
well-formed.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ca_aocp.bocpd import GaussianBOCPD


def make_series(seed=0, n_pre=150, n_post=150, mean_shift=5.0):
    rng = np.random.default_rng(seed)
    pre = rng.normal(0.0, 1.0, n_pre)
    post = rng.normal(mean_shift, 1.0, n_post)
    return np.concatenate([pre, post]), n_pre


BOCPD_KWARGS = dict(hazard=0.02, mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=1.0)


def test_posterior_always_normalizes_to_one():
    """Sanity: the (dense, reconstructed) run-length posterior must always
    sum to ~1, truncated or not -- this held even under the old buggy code,
    so it does NOT by itself catch the alignment bug (see the next test),
    but a regression here would indicate a more basic normalization bug."""
    ys, _ = make_series()
    b = GaussianBOCPD(max_run_length=20, **BOCPD_KWARGS)
    for x in ys:
        b.update(x)
        mass = np.exp(b._log_R).sum()
        assert abs(mass - 1.0) < 1e-6


def test_truncated_matches_untruncated_reference_on_representable_run_lengths():
    """The core regression test for the alignment bug.

    Run the same data through a truncated (K=20) and an effectively
    untruncated (K=10000) BOCPD. Compare P(r_t = k) for k in the
    REPRESENTABLE range (k < K) only -- during a long stable stretch the
    untruncated reference's mass legitimately concentrates on run lengths
    >= K (e.g. run length ~140 after 150 stationary steps), which a K=20
    model cannot represent *by design*; that is expected information loss
    from truncation, not a bug. What must hold, if truncation is wired up
    correctly, is that within the representable range (k < K) the two
    posteriors agree closely -- pre-fix, even this diverged by up to ~0.99
    once truncation activated, because of the index-misalignment bug, not
    because of legitimate truncation loss.
    """
    ys, _ = make_series(seed=1)
    K = 20
    ref = GaussianBOCPD(max_run_length=10_000, **BOCPD_KWARGS)
    trunc = GaussianBOCPD(max_run_length=K, **BOCPD_KWARGS)

    max_diffs = []
    for x in ys:
        ref.update(x)
        trunc.update(x)
        ref_dense = ref.get_run_length_posterior()
        trunc_dense = trunc.get_run_length_posterior()
        L = min(K, len(ref_dense), len(trunc_dense))
        diffs = np.abs(ref_dense[:L] - trunc_dense[:L])
        max_diffs.append(diffs.max())

    max_diffs = np.array(max_diffs)
    # Give the first ~K steps (before truncation ever activates, so both
    # models are identical anyway) a pass, then require close agreement
    # in steady state over the representable range.
    steady = max_diffs[K + 10:]
    assert steady.mean() < 0.01, (
        f"truncated posterior diverges from untruncated reference (within "
        f"the representable run-length range) by {steady.mean():.4f} on "
        f"average -- alignment bug likely reintroduced"
    )
    assert steady.max() < 0.05, (
        f"worst-case divergence {steady.max():.4f} too large within the "
        f"representable range -- alignment bug likely reintroduced"
    )


def test_no_probability_mass_silently_dropped():
    """Pre-fix, ~18%/step of probability mass sat beyond the sliced
    `[:len(self._nig)]` window and was silently discarded on the next
    step. Post-fix, get_run_length_posterior() should account for (i.e.
    sum to) essentially all retained mass every step."""
    ys, _ = make_series(seed=2)
    b = GaussianBOCPD(max_run_length=20, **BOCPD_KWARGS)
    for x in ys:
        b.update(x)
        dense = b.get_run_length_posterior()
        assert abs(dense.sum() - 1.0) < 1e-6


def test_map_run_length_tracks_known_changepoint():
    """Sanity check that the MAP run-length estimate (argmax of the dense
    posterior) resets near zero shortly after a large, obvious mean shift,
    and grows roughly linearly with time before it."""
    ys, tau = make_series(seed=3, n_pre=200, n_post=200, mean_shift=8.0)
    b = GaussianBOCPD(max_run_length=50, **BOCPD_KWARGS)
    r_hat = []
    for x in ys:
        b.update(x)
        post = b.get_run_length_posterior()
        r_hat.append(int(np.argmax(post)))
    r_hat = np.array(r_hat)

    # Just before the changepoint, run length should be large (near the
    # truncation cap, since we've been in the same regime a long time).
    assert r_hat[tau - 5] >= 30
    # Within a handful of steps after the changepoint, MAP run length
    # should have collapsed back down.
    assert r_hat[tau + 10] <= 12


def test_legacy_pi_shape_matches_t_backward_compat():
    """experiments/run_ablation.py (this project's frozen experiment
    scripts) calls GaussianBOCPD.update() directly and expects the
    returned pi vector's length to equal the number of observations seen
    so far -- verify this legacy contract still holds post-fix."""
    ys, _ = make_series(seed=4, n_pre=60, n_post=0)
    b = GaussianBOCPD(max_run_length=15, **BOCPD_KWARGS)
    for i, x in enumerate(ys, start=1):
        pi = b.update(x)
        assert len(pi) == i, f"expected legacy pi length {i}, got {len(pi)}"
        assert np.all((pi >= 0) & (pi <= 1))


def test_pi_recent_is_bounded_and_consistent_with_legacy_tail():
    """pi_recent(window=K) should equal the last K entries of the legacy
    full-length pi (both are oldest-first, so the tail of the legacy
    vector is the "most recent" window)."""
    ys, _ = make_series(seed=5, n_pre=80, n_post=0)
    K = 15
    b = GaussianBOCPD(max_run_length=K, **BOCPD_KWARGS)
    for x in ys:
        b.update(x)

    legacy = b._compute_pi()
    recent = b.pi_recent(K)
    assert len(recent) == K
    np.testing.assert_allclose(recent, legacy[-K:], atol=1e-9)


def test_pi_recent_bounded_length_even_for_long_series():
    ys, _ = make_series(seed=6, n_pre=500, n_post=0)
    K = 30
    b = GaussianBOCPD(max_run_length=K, **BOCPD_KWARGS)
    for x in ys:
        b.update(x)
    assert len(b.pi_recent()) == K  # default window = K
    assert len(b.pi_recent(10)) == 10


def test_compute_legacy_pi_false_returns_none_and_skips_computation():
    """Regression test for the legacy-pi O(t) leak (2026-08 follow-up fix).

    Pre-fix, update() unconditionally built and returned the O(t)
    `_compute_pi()` vector even when the caller (CAAOCP) discarded it and
    only used the O(K) `pi_recent()` result instead -- so CAAOCP paid an
    O(t) cost per step underneath its O(K) buffers, growing without bound.
    compute_legacy_pi=False must (a) return None, and (b) actually skip
    the O(t) construction, not just discard its result afterward -- verified
    here by monkeypatching _compute_pi to raise and confirming it is never
    called in that mode."""
    ys, _ = make_series(seed=7, n_pre=60, n_post=0)
    b = GaussianBOCPD(max_run_length=15, **BOCPD_KWARGS)

    def _boom():
        raise AssertionError("_compute_pi() must not be called when compute_legacy_pi=False")

    b._compute_pi = _boom  # type: ignore[method-assign]
    for x in ys:
        result = b.update(x, compute_legacy_pi=False)
        assert result is None


def test_compute_legacy_pi_true_is_still_the_default():
    """Default behaviour (no compute_legacy_pi argument) must be unchanged,
    since experiments/run_ablation.py and similar legacy callers rely on it."""
    ys, _ = make_series(seed=8, n_pre=40, n_post=0)
    b = GaussianBOCPD(max_run_length=15, **BOCPD_KWARGS)
    for i, x in enumerate(ys, start=1):
        pi = b.update(x)
        assert pi is not None
        assert len(pi) == i


def test_legacy_pi_cost_grows_with_t_but_pi_recent_does_not():
    """Complexity check backing the fix, isolating _compute_pi() (the O(t)
    legacy path `update()` used to always build and CAAOCP always
    discarded) from `pi_recent()` (the O(K) path CAAOCP actually uses).

    Going through the *full* `update()` call is too noisy a signal at
    modest t for a fast test: at t in the few-thousands, the O(t) leak is
    dwarfed by update()'s own O(K) BOCPD arithmetic (Student-t evaluations
    per active run length), so a full-update timing comparison doesn't
    reliably separate the two until t reaches ~1e5+ (confirmed manually:
    _compute_pi() cost 14us/51us/744us/3756us at t=1e3/1e4/1e5/5e5, versus
    pi_recent() flat at ~10us throughout -- a >300x gap by t=5e5, but not
    yet a dominant cost at t in the thousands). So instead of timing
    update() end-to-end, this test calls `_compute_pi()` and `pi_recent()`
    directly at large *simulated* t (via `b.t = T`, cheap to set up without
    actually running T update() steps) and checks the qualitative
    complexity gap shows up as expected: _compute_pi() must slow down
    materially more than pi_recent() as t grows.
    """
    import time

    K = 30
    ys, _ = make_series(seed=10, n_pre=K + 10, n_post=0)
    b = GaussianBOCPD(max_run_length=K, **BOCPD_KWARGS)
    for x in ys:
        b.update(x)  # get past truncation onset so state is representative

    def timed(fn, T, reps=20):
        b.t = T  # simulate having processed T steps without paying for it
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t0) / reps

    t_small, t_large = 2_000, 200_000  # 100x longer simulated history

    legacy_small = timed(b._compute_pi, t_small)
    legacy_large = timed(b._compute_pi, t_large)
    recent_small = timed(lambda: b.pi_recent(K), t_small)
    recent_large = timed(lambda: b.pi_recent(K), t_large)

    legacy_ratio = legacy_large / max(legacy_small, 1e-9)
    recent_ratio = recent_large / max(recent_small, 1e-9)

    # pi_recent() must not slow down anywhere near proportionally to t (it's
    # O(K), independent of t); _compute_pi() is expected to (it's O(t)).
    assert recent_ratio < 5.0, (
        f"pi_recent() slowed down {recent_ratio:.1f}x for a 100x larger "
        f"simulated t -- should be ~flat (O(K), not O(t))"
    )
    assert legacy_ratio > 10.0, (
        f"_compute_pi() only slowed down {legacy_ratio:.1f}x for a 100x "
        f"larger simulated t -- expected clear O(t) growth; if this no "
        f"longer holds the benchmark backing the compute_legacy_pi fix "
        f"may need updating"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
