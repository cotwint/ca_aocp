"""
Multi-Seed Robustness Check for the Component Ablation
=========================================================
Repeats the 8-method ablation (see run_ablation.py) across many random
seeds on both a Synthetic-CP-style series and a QC-style series, to check
whether the single-seed findings (esp. "hard-reset ties soft BOCPD
weighting") are robust.

Design notes / caveats (please read before trusting the numbers):
  - Both datasets are generated from the paper's *reported moments*
    (Table 8), not the original data (QC's original series isn't released
    with the paper). This is therefore a robustness check on a matched
    generative model, not a replication of the exact paper numbers.
  - Writes incrementally to a CSV (append mode) so the script can be run
    in seed-range batches across multiple sandboxed shell calls without
    losing progress.
  - Definitions used for the two "process" diagnostics (not standard, our
    own operationalisation -- state this if quoting in the paper):
      * false_alarm_rate: fraction of PRE-change steps (t <= tau*, t > K)
        where BOCPD's MAP run-length estimate r_hat(t) <= 2, i.e. the
        detector spuriously "believes" a changepoint just happened even
        though the regime hasn't changed.
      * cp_delay: first post-change step (t > tau*) at which r_hat(t) <= 2
        AND stays <= 5 for the next 5 steps, minus tau*. NaN if this never
        happens within the series (censored / non-detection).
      * recovery_step: first n (steps after tau*) such that the trailing
        10-step rolling coverage from tau*+n onward is >= 1-alpha-0.1
        (i.e. >=0.8 for alpha=0.1). NaN if never reached.
    These are heuristic operationalisations for this ablation only; they
    are NOT the formal quantities used in the paper's theorems.

Usage:
    python3 experiments/run_multiseed.py --seed-start 0 --seed-end 3 \
        --out experiments/output_multiseed/raw.csv
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_ablation import make_methods, winkler_score


# ═══════════════════════════════════════════════════════════════════════
# Parametric dataset generators (paper Table 8 moments)
# ═══════════════════════════════════════════════════════════════════════

def gen_synthetic_cp(seed: int) -> tuple[np.ndarray, int]:
    """T=1000, tau*=500, N(0.01,0.98^2) -> N(4.04,0.97^2)."""
    rng = np.random.default_rng(seed)
    T, tau = 1000, 500
    pre = rng.normal(0.01, 0.98, tau)
    post = rng.normal(4.04, 0.97, T - tau)
    return np.concatenate([pre, post]), tau


def gen_qc(seed: int) -> tuple[np.ndarray, int]:
    """T=313, tau*=148, N(0.67,1.16^2) -> N(4.33,1.01^2) + drift 0.009/step."""
    rng = np.random.default_rng(seed)
    T, tau = 313, 148
    pre = rng.normal(0.67, 1.16, tau)
    n_post = T - tau
    drift = 0.009 * np.arange(1, n_post + 1)
    post = rng.normal(4.33, 1.01, n_post) + drift
    return np.concatenate([pre, post]), tau


DATASETS = {
    "Synthetic-CP": dict(gen=gen_synthetic_cp, hazard=0.005, pred_window=20),
    "Quality-Control": dict(gen=gen_qc, hazard=0.010, pred_window=15),
}


# ═══════════════════════════════════════════════════════════════════════
# Per-run diagnostics
# ═══════════════════════════════════════════════════════════════════════

def rolling_coverage(covered: np.ndarray, window: int = 10) -> np.ndarray:
    c = covered.astype(float)
    if len(c) < window:
        return np.array([])
    return np.convolve(c, np.ones(window) / window, mode="valid")


def recovery_step(covered: np.ndarray, tau: int, alpha: float, window: int = 10, target: float = None) -> float:
    target = target if target is not None else (1 - alpha - 0.1)
    post = covered[tau:]
    roll = rolling_coverage(post, window)
    hits = np.where(roll >= target)[0]
    return float(hits[0]) if len(hits) > 0 else np.nan


def false_alarm_rate(r_hat: np.ndarray, tau: int, K: int) -> float:
    pre = r_hat[K:tau]
    if len(pre) == 0 or np.all(np.isnan(pre)):
        return np.nan
    return float(np.nanmean(pre <= 2))


def cp_delay(r_hat: np.ndarray, tau: int, T: int) -> float:
    for t in range(tau, min(T - 5, T)):
        window = r_hat[t: t + 5]
        if len(window) == 5 and np.all(window <= 5) and r_hat[t] <= 2:
            return float(t - tau)
    return np.nan


def summarize_run(name, res, ys, tau, alpha, K, uses_bocpd):
    cov = res["covered"].astype(float)
    lo, hi = res["lower"], res["upper"]
    w = winkler_score(ys, lo, hi, alpha)
    T = len(ys)

    row = {
        "Method": name,
        "coverage": cov.mean(),
        "cov_gap": abs(cov.mean() - (1 - alpha)),
        "mean_width": res["width"].mean(),
        "winkler": w.mean(),
        "recovery_step": recovery_step(cov.astype(int), tau, alpha),
    }
    for h in (10, 25, 50):
        end = min(tau + h, T)
        seg = cov[tau:end]
        row[f"postcov{h}"] = seg.mean() if len(seg) > 0 else np.nan

    if uses_bocpd:
        r_hat = res["r_hat"]
        row["false_alarm_rate"] = false_alarm_rate(r_hat, tau, K)
        row["cp_delay"] = cp_delay(r_hat, tau, T)
    else:
        row["false_alarm_rate"] = np.nan
        row["cp_delay"] = np.nan

    return row


# ═══════════════════════════════════════════════════════════════════════
# Main batch runner
# ═══════════════════════════════════════════════════════════════════════

USES_BOCPD = {
    "ACI": False, "EWMA-ACI": False,
    "BOCPD-weighted CP": True, "BOCPD-weighted ACI": True,
    "CA-AOCP w/o ACI": True, "Full CA-AOCP": True,
    "Hard-reset CP": True, "Hard-reset ACI": True,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-start", type=int, required=True)
    ap.add_argument("--seed-end", type=int, required=True)  # exclusive
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--datasets", type=str, default="Synthetic-CP,Quality-Control")
    args = ap.parse_args()

    alpha = 0.1
    eta = 0.02
    temperature = 0.5
    decay = 0.01
    bocpd_kwargs = dict(mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=1.0)
    K = 50
    alpha_min, alpha_max = alpha / 2, (1 + alpha) / 2
    init_radius = 1.0

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()

    ds_names = args.datasets.split(",")
    t_start = time.perf_counter()
    rows = []

    for seed in range(args.seed_start, args.seed_end):
        for ds_name in ds_names:
            cfg = DATASETS[ds_name]
            ys, tau = cfg["gen"](seed)
            methods = make_methods(
                alpha, eta, temperature, decay, cfg["hazard"], bocpd_kwargs, K,
                cfg["pred_window"], init_radius, alpha_min, alpha_max,
            )
            for mname, m in methods.items():
                res = m.run(ys)
                row = summarize_run(mname, res, ys, tau, alpha, K, USES_BOCPD[mname])
                row["Dataset"] = ds_name
                row["seed"] = seed
                rows.append(row)

        elapsed = time.perf_counter() - t_start
        print(f"seed {seed} done ({elapsed:.1f}s elapsed)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, mode="a", header=write_header, index=False)
    print(f"Appended {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
