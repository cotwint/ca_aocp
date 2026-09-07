"""
Shift Grid: Where (if anywhere) does soft BOCPD weighting beat hard-reset?
============================================================================
The multi-seed ablation (run_multiseed.py, 30 seeds x 2 datasets) found NO
statistically distinguishable advantage of soft BOCPD posterior weighting
over a simple hard-reset (MAP changepoint + uniform window) baseline, on
either a sharp abrupt-mean-shift Synthetic-CP-style series or a QC-style
series. Both of those series have a strong, instantaneous, easy-to-detect
changepoint (Delta_mu/sigma ~= 3-4).

Hypothesis: soft weighting should differentiate from hard-reset precisely
when the BOCPD run-length MAP estimate becomes unstable/noisy -- i.e. when
the changepoint signal is WEAK or AMBIGUOUS. Two ways to make it weak:
  (a) smaller mean shift (lower Delta_mu/sigma)
  (b) a gradual ramp into the new regime instead of an instantaneous step
      (the "changepoint" is smeared over `ramp_len` steps)
We also include a no-changepoint (stationary) control to check which
weighting scheme is more ROBUST to false alarms (spurious resets on pure
noise), since a hard reset throws away all history on every false alarm
while soft weighting only down-weights it.

Grid: shift in {1, 2, 4} sigma  x  ramp_len in {0, 10, 40} steps, all at
T=300, tau*=100, sigma=1, hazard=0.01 (fixed across the whole grid -- NOT
tuned per condition, so as not to leak changepoint-strength information
into the detector's prior). Plus one no-changepoint control series.

Only the 4 methods that isolate the soft-vs-hard question are run:
  BOCPD-weighted CP / ACI   (soft posterior weight, decay=0)
  Hard-reset CP / ACI       (MAP changepoint reset, uniform window)
(CA-AOCP-w/o-ACI and Full CA-AOCP were already shown in the ablation to
be numerically indistinguishable from the "no-decay" BOCPD-weighted
variants, so they are dropped here to save compute.)

In addition to the earlier metrics, this script tracks a MECHANISM
diagnostic: the effective calibration-set size used to form q_t at fixed
offsets after tau* -- N_eff = 1/sum(w~^2) for soft weighting, and the
literal window size (r_hat+1) for hard-reset. This is meant to make
VISIBLE *why* the two schemes might differ even when final coverage looks
similar (e.g. hard-reset calibrating off 1-2 points right after a reset).

Usage:
    python3 experiments/run_shift_grid.py --seed-start 0 --seed-end 5 \
        --out experiments/output_shiftgrid/raw.csv
"""

from __future__ import annotations
import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ca_aocp.bocpd import GaussianBOCPD
from ca_aocp.conformal import weighted_quantile, prediction_interval, coverage_indicator
from ca_aocp.adaptive_update import smooth_surrogate, adaptive_update
from ca_aocp.predictors import RollingMeanPredictor
from experiments.run_ablation import winkler_score
from experiments.run_multiseed import recovery_step, false_alarm_rate, cp_delay


# ═══════════════════════════════════════════════════════════════════════
# Data generator with configurable ramp
# ═══════════════════════════════════════════════════════════════════════

def gen_ramped_series(seed: int, T: int, tau: int, shift_sigma: float,
                       ramp_len: int, sigma: float = 1.0, pre_mean: float = 0.0) -> np.ndarray:
    """Piecewise mean-shift series with a linear ramp of length `ramp_len`.

    ramp_len=0 -> instantaneous step at tau (as in the paper's datasets).
    ramp_len=None or shift_sigma=0 -> stationary (no true changepoint), used
    as the false-alarm robustness control.
    """
    rng = np.random.default_rng(seed)
    post_mean = pre_mean + shift_sigma * sigma
    means = np.full(T, pre_mean)
    if shift_sigma != 0:
        if ramp_len <= 0:
            means[tau:] = post_mean
        else:
            ramp_end = min(tau + ramp_len, T)
            n_ramp = ramp_end - tau
            means[tau:ramp_end] = np.linspace(pre_mean, post_mean, n_ramp, endpoint=False)
            means[ramp_end:] = post_mean
    return means + rng.normal(0, sigma, T)


# ═══════════════════════════════════════════════════════════════════════
# Method (reuses same math as GenericMethod, but also tracks the
# calibration-set "effective size" mechanism diagnostic)
# ═══════════════════════════════════════════════════════════════════════

class TrackedMethod:
    def __init__(self, alpha, eta, temperature, hazard, bocpd_kwargs, K,
                 hard_reset, adapt_alpha, pred_window, init_radius, alpha_min, alpha_max):
        self.alpha = alpha
        self.eta = eta
        self.temperature = temperature
        self.hard_reset = hard_reset
        self.adapt_alpha = adapt_alpha
        self.alpha_min, self.alpha_max = alpha_min, alpha_max
        self._alpha_t = alpha
        self._q_t = init_radius
        self.K = K
        # (2026-08 follow-up) All four methods here are BOCPD-driven
        # (soft-weighted or hard-reset), so all four are meant to be
        # K-truncated per the paper's CA-AOCP hyperparameter table. Before
        # this fix, self._scores was an unbounded list matched against the
        # legacy full-length pi from GaussianBOCPD.update()'s old return
        # value -- so this script's soft-vs-hard-reset comparison (the whole
        # point of run_shift_grid.py) was never actually run under K
        # truncation, only under BOCPD's internal K-truncated run-length
        # posterior extended (via legacy _compute_pi()) over the full,
        # untruncated score history.
        self._scores: deque[float] = deque(maxlen=K)
        self._predictor = RollingMeanPredictor(window=pred_window)
        self._bocpd = GaussianBOCPD(hazard=hazard, max_run_length=K, **bocpd_kwargs)

    def _weights_and_neff(self, weight_info, n):
        if self.hard_reset:
            r_hat = int(np.argmax(weight_info))
            window = min(r_hat + 1, n)
            w = np.zeros(n)
            w[-window:] = 1.0 / window
            neff = float(window)  # effective size = window size (uniform)
        else:
            log_pi = np.log(np.clip(weight_info, 1e-300, None))
            log_pi -= log_pi.max()
            w = np.exp(log_pi)
            w /= w.sum()
            neff = 1.0 / np.dot(w, w)
        return w, neff

    def run(self, ys: np.ndarray) -> dict:
        T = len(ys)
        covered = np.zeros(T, dtype=int)
        widths = np.zeros(T)
        r_hat_seq = np.full(T, np.nan)
        neff_seq = np.full(T, np.nan)

        for t_idx in range(T):
            y_t = float(ys[t_idx])
            yhat = self._predictor.predict(None)
            lo, hi = prediction_interval(yhat, self._q_t)
            covered[t_idx] = coverage_indicator(y_t, lo, hi)
            widths[t_idx] = hi - lo

            s_t = abs(y_t - yhat)
            self._scores.append(s_t)

            ell_t = smooth_surrogate(s_t, self._q_t, self.temperature)
            if self.adapt_alpha:
                alpha_next = adaptive_update(self._alpha_t, ell_t, self.alpha, self.eta,
                                              self.alpha_min, self.alpha_max)
            else:
                alpha_next = self.alpha

            # compute_legacy_pi=False: avoid the O(t) legacy-pi leak (same
            # fix as ca_aocp.algorithm.CAAOCP); we only use the bounded
            # pi_recent()/get_run_length_posterior() views below.
            self._bocpd.update(s_t, compute_legacy_pi=False)
            run_post = self._bocpd.get_run_length_posterior()
            r_hat_seq[t_idx] = int(np.argmax(run_post))
            weight_info = run_post if self.hard_reset else self._bocpd.pi_recent(self.K)

            n = len(self._scores)
            if n >= 2:
                w, neff = self._weights_and_neff(weight_info, n)
                neff_seq[t_idx] = neff
                scores_arr = np.array(self._scores)
                self._q_t = weighted_quantile(scores_arr, w, alpha_next)

            self._alpha_t = alpha_next
            self._predictor.update(None, y_t)

        return {"covered": covered, "width": widths, "r_hat": r_hat_seq, "neff": neff_seq}


def make_grid_methods(alpha, eta, temperature, hazard, bocpd_kwargs, K,
                       pred_window, init_radius, alpha_min, alpha_max):
    common = dict(alpha=alpha, eta=eta, temperature=temperature, hazard=hazard,
                  bocpd_kwargs=bocpd_kwargs, K=K, pred_window=pred_window,
                  init_radius=init_radius, alpha_min=alpha_min, alpha_max=alpha_max)
    return {
        "BOCPD-weighted CP":  TrackedMethod(hard_reset=False, adapt_alpha=False, **common),
        "BOCPD-weighted ACI": TrackedMethod(hard_reset=False, adapt_alpha=True,  **common),
        "Hard-reset CP":      TrackedMethod(hard_reset=True,  adapt_alpha=False, **common),
        "Hard-reset ACI":     TrackedMethod(hard_reset=True,  adapt_alpha=True,  **common),
    }


# ═══════════════════════════════════════════════════════════════════════
# Per-run summary
# ═══════════════════════════════════════════════════════════════════════

def summarize(name, res, ys, tau, alpha, K):
    cov = res["covered"].astype(float)
    T = len(ys)
    row = {
        "Method": name,
        "coverage": cov.mean(),
        "cov_gap": abs(cov.mean() - (1 - alpha)),
        "mean_width": res["width"].mean(),
        "recovery_step": recovery_step(cov.astype(int), tau, alpha),
        "false_alarm_rate": false_alarm_rate(res["r_hat"], tau, K),
        "cp_delay": cp_delay(res["r_hat"], tau, T),
    }
    for h in (10, 25, 50, 100):
        end = min(tau + h, T)
        seg = cov[tau:end]
        w_seg = res["width"][tau:end]
        neff_seg = res["neff"][tau:end]
        row[f"postcov{h}"] = seg.mean() if len(seg) else np.nan
        row[f"postwidth{h}"] = w_seg.mean() if len(w_seg) else np.nan
    # mechanism: effective calibration size right after the change
    for off in (1, 5, 10, 25):
        idx = tau + off
        row[f"neff_t+{off}"] = res["neff"][idx] if idx < T else np.nan
    # overshoot: worst (max) width in first 25 post-change steps, relative
    # to the median width in the last 30 pre-change steps (stable regime)
    pre_stable = res["width"][max(0, tau - 30):tau]
    post_early = res["width"][tau: min(tau + 25, T)]
    if len(pre_stable) > 0 and len(post_early) > 0 and np.median(pre_stable) > 1e-9:
        row["width_overshoot_ratio"] = float(post_early.max() / np.median(pre_stable))
    else:
        row["width_overshoot_ratio"] = np.nan
    return row


# ═══════════════════════════════════════════════════════════════════════
# Grid definition
# ═══════════════════════════════════════════════════════════════════════

T, TAU = 300, 100
SHIFTS = [1.0, 2.0, 4.0]
RAMPS = [0, 10, 40]


def iter_conditions():
    for shift in SHIFTS:
        for ramp in RAMPS:
            yield f"shift{shift}_ramp{ramp}", shift, ramp
    yield "no_changepoint_control", 0.0, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-start", type=int, required=True)
    ap.add_argument("--seed-end", type=int, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    alpha = 0.1
    eta = 0.02
    temperature = 0.5
    hazard = 0.01  # fixed across the whole grid -- not tuned to shift/ramp
    bocpd_kwargs = dict(mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=1.0)
    K = 50
    alpha_min, alpha_max = alpha / 2, (1 + alpha) / 2
    init_radius = 1.0
    pred_window = 15

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()

    rows = []
    t0 = time.perf_counter()
    for seed in range(args.seed_start, args.seed_end):
        for cond_name, shift, ramp in iter_conditions():
            ys = gen_ramped_series(seed, T, TAU, shift, ramp)
            methods = make_grid_methods(alpha, eta, temperature, hazard, bocpd_kwargs, K,
                                         pred_window, init_radius, alpha_min, alpha_max)
            for mname, m in methods.items():
                res = m.run(ys)
                row = summarize(mname, res, ys, TAU, alpha, K)
                row["condition"] = cond_name
                row["shift"] = shift
                row["ramp"] = ramp
                row["seed"] = seed
                rows.append(row)
        print(f"seed {seed} done ({time.perf_counter()-t0:.1f}s elapsed)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, mode="a", header=write_header, index=False)
    print(f"Appended {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
