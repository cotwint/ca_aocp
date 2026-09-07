"""
Hyperparameter Sensitivity: hazard h, decay lambda, truncation depth K
=========================================================================
One-factor-at-a-time (OFAT) sweep around the paper's default
(h=0.005, lambda=0.01, K=50), on Full CA-AOCP, using the Synthetic-CP
generator (Table 8 moments: T=1000, tau*=500). Each config is run over
multiple seeds to get mean +/- SE, not a single point estimate.

Also runs a "mismatched hazard" check: the paper uses h=0.005 for
Synthetic-CP and h=0.01 for QC, i.e. a *different* hazard tuned per
dataset. This script additionally cross-runs each dataset with the
*other* dataset's hazard, to quantify how much is lost if hazard is not
tuned using knowledge of the true regime length (addresses the "does
hyperparameter choice leak changepoint-location information" concern).

Usage:
    python3 experiments/run_hparam_sensitivity.py --sweep hazard --seed-start 0 --seed-end 10 \
        --out experiments/output_hparam/raw.csv
    python3 experiments/run_hparam_sensitivity.py --sweep decay ...
    python3 experiments/run_hparam_sensitivity.py --sweep K ...
    python3 experiments/run_hparam_sensitivity.py --sweep cross_hazard ...
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

from experiments.run_ablation import GenericMethod, winkler_score
from experiments.run_multiseed import gen_synthetic_cp, gen_qc, recovery_step


DEFAULTS = dict(hazard=0.005, decay=0.01, K=50)

SWEEPS = {
    "hazard": [0.002, 0.005, 0.01, 0.02, 0.05],
    "decay":  [0.0, 0.005, 0.01, 0.02, 0.05],
    "K":      [20, 50, 100, 200],
}


def summarize(res, ys, tau, alpha):
    cov = res["covered"].astype(float)
    lo, hi = res["lower"], res["upper"]
    w = winkler_score(ys, lo, hi, alpha)
    row = {
        "coverage": cov.mean(),
        "cov_gap": abs(cov.mean() - (1 - alpha)),
        "mean_width": res["width"].mean(),
        "winkler": w.mean(),
        "recovery_step": recovery_step(cov.astype(int), tau, alpha),
    }
    for h in (10, 25, 50):
        end = min(tau + h, len(ys))
        seg = cov[tau:end]
        row[f"postcov{h}"] = seg.mean() if len(seg) else np.nan
    return row


def run_one(ys, tau, alpha, eta, temperature, hazard, decay, K, pred_window, init_radius, alpha_min, alpha_max):
    m = GenericMethod(
        alpha=alpha, eta=eta, temperature=temperature, decay=decay, hazard=hazard,
        bocpd_kwargs=dict(mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=1.0),
        max_run_length=K, use_bocpd=True, adapt_alpha=True, hard_reset=False,
        predictor=__import__("ca_aocp.predictors", fromlist=["RollingMeanPredictor"]).RollingMeanPredictor(window=pred_window),
        init_radius=init_radius, alpha_min=alpha_min, alpha_max=alpha_max,
    )
    res = m.run(ys)
    return summarize(res, ys, tau, alpha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", choices=["hazard", "decay", "K", "cross_hazard"], required=True)
    ap.add_argument("--seed-start", type=int, required=True)
    ap.add_argument("--seed-end", type=int, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    alpha, eta, temperature = 0.1, 0.02, 0.5
    alpha_min, alpha_max = alpha / 2, (1 + alpha) / 2
    init_radius = 1.0

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()

    rows = []
    t0 = time.perf_counter()

    if args.sweep == "cross_hazard":
        # Synthetic-CP with its own h=0.005 vs QC's h=0.01; and vice versa for QC.
        configs = [
            ("Synthetic-CP", gen_synthetic_cp, 0.005, "matched", 20),
            ("Synthetic-CP", gen_synthetic_cp, 0.010, "mismatched(QC's h)", 20),
            ("Quality-Control", gen_qc, 0.010, "matched", 15),
            ("Quality-Control", gen_qc, 0.005, "mismatched(Synth's h)", 15),
        ]
        for seed in range(args.seed_start, args.seed_end):
            for ds_name, gen_fn, hazard, tag, pw in configs:
                ys, tau = gen_fn(seed)
                row = run_one(ys, tau, alpha, eta, temperature, hazard, DEFAULTS["decay"], DEFAULTS["K"],
                               pw, init_radius, alpha_min, alpha_max)
                row.update(dict(sweep="cross_hazard", dataset=ds_name, hazard=hazard, tag=tag,
                                 decay=DEFAULTS["decay"], K=DEFAULTS["K"], seed=seed))
                rows.append(row)
            print(f"seed {seed} done ({time.perf_counter()-t0:.1f}s)", flush=True)
    else:
        values = SWEEPS[args.sweep]
        for seed in range(args.seed_start, args.seed_end):
            ys, tau = gen_synthetic_cp(seed)
            for v in values:
                cfg = dict(DEFAULTS)
                cfg[args.sweep] = v
                row = run_one(ys, tau, alpha, eta, temperature, cfg["hazard"], cfg["decay"], cfg["K"],
                               20, init_radius, alpha_min, alpha_max)
                row.update(dict(sweep=args.sweep, dataset="Synthetic-CP", hazard=cfg["hazard"],
                                 decay=cfg["decay"], K=cfg["K"], swept_value=v, seed=seed))
                rows.append(row)
            print(f"seed {seed} done ({time.perf_counter()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    # Fixed, explicit column order -- prevents silent column misalignment when
    # different sweep types (different row-dict key sets/orders) are appended
    # to the same CSV via mode="a" (pandas does NOT reconcile headers on append).
    fixed_cols = ["coverage", "cov_gap", "mean_width", "winkler", "recovery_step",
                  "postcov10", "postcov25", "postcov50", "sweep", "dataset",
                  "hazard", "decay", "K", "swept_value", "tag", "seed"]
    for c in fixed_cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[fixed_cols]
    df.to_csv(out_path, mode="a", header=write_header, index=False)
    print(f"Appended {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
