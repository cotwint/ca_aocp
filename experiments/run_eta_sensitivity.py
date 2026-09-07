"""
ACI Learning-Rate (eta) Sensitivity + Redundancy Test
=========================================================
Motivated by a close read of Gibbs & Candes (2021), "Adaptive Conformal
Inference Under Distribution Shift":

  - ACI's coverage guarantee is a LONG-RUN average (Prop 4.1: the bias term
    is O(1/T)). Short, single-changepoint series may sit in an unfavorable
    finite-sample regime regardless of whether ACI is "helping" in any
    fundamental sense.
  - The optimal step size (their gamma, our eta) scales with
    sqrt(E|alpha*_{t+1} - alpha*_t|) (Theorem 4.2): too large a step size
    causes alpha_t to oscillate and inflates cov_gap without buying faster
    recovery. The paper's own gamma=0.005 was hand-tuned to their specific
    volatility dataset -- nothing guarantees our fixed eta=0.02 default is
    well matched to Synthetic-CP / QC's single, rare shift.
  - ACI's marginal value is largest when the base conformity score is NOT
    otherwise being re-stabilized (their Section 5 score-stationarity
    result: an unnormalized, still-nonstationary score benefits far more
    from ACI than an already-renormalized one). Since BOCPD-weighting
    already re-bases the score distribution at every regime, ACI's job on
    top of BOCPD may be largely redundant -- predicting Full CA-AOCP has
    LESS to gain from any eta than plain ACI (no BOCPD) does.

Two sub-experiments, both multi-seed, both on Synthetic-CP and QC:

  (A) eta_full        OFAT sweep of eta on Full CA-AOCP alone
                       (BOCPD + decay + ACI), to see whether the cov_gap
                       penalty documented in Task #5 shrinks as eta shrinks.

  (B) eta_redundancy   For each seed and each eta, run BOTH plain "ACI"
                       (no BOCPD) and "Full CA-AOCP" (BOCPD + decay + ACI)
                       on the identical data draw, to check whether there is
                       ANY eta at which Full CA-AOCP beats plain ACI
                       (revisits the "beats plain ACI in only 2/7 real
                       episodes" finding from run_real_segmentation.py).

Usage:
    python3 experiments/run_eta_sensitivity.py --sweep eta_full --seed-start 0 --seed-end 10 --out experiments/output_eta/raw.csv
    python3 experiments/run_eta_sensitivity.py --sweep eta_redundancy --seed-start 0 --seed-end 10 --out experiments/output_eta/raw.csv
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
from ca_aocp.predictors import RollingMeanPredictor


ETA_GRID = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

DATASETS = {
    "Synthetic-CP":    dict(gen=gen_synthetic_cp, hazard=0.005, pred_window=20),
    "Quality-Control": dict(gen=gen_qc,            hazard=0.010, pred_window=15),
}

DECAY = 0.01
K = 50
ALPHA = 0.1
TEMPERATURE = 0.5
INIT_RADIUS = 1.0
ALPHA_MIN, ALPHA_MAX = ALPHA / 2, (1 + ALPHA) / 2
BOCPD_KWARGS = dict(mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=1.0)


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


def run_full(ys, tau, eta, hazard, pred_window):
    m = GenericMethod(
        alpha=ALPHA, eta=eta, temperature=TEMPERATURE, decay=DECAY, hazard=hazard,
        bocpd_kwargs=BOCPD_KWARGS, max_run_length=K,
        use_bocpd=True, adapt_alpha=True, hard_reset=False,
        predictor=RollingMeanPredictor(window=pred_window),
        init_radius=INIT_RADIUS, alpha_min=ALPHA_MIN, alpha_max=ALPHA_MAX,
    )
    res = m.run(ys)
    return summarize(res, ys, tau, ALPHA)


def run_plain_aci(ys, tau, eta, pred_window):
    m = GenericMethod(
        alpha=ALPHA, eta=eta, temperature=TEMPERATURE, decay=0.0, hazard=0.01,
        bocpd_kwargs=BOCPD_KWARGS, max_run_length=K,
        use_bocpd=False, adapt_alpha=True, hard_reset=False,
        predictor=RollingMeanPredictor(window=pred_window),
        init_radius=INIT_RADIUS, alpha_min=ALPHA_MIN, alpha_max=ALPHA_MAX,
    )
    res = m.run(ys)
    return summarize(res, ys, tau, ALPHA)


FIXED_COLS = ["coverage", "cov_gap", "mean_width", "winkler", "recovery_step",
              "postcov10", "postcov25", "postcov50", "sweep", "dataset",
              "eta", "method", "seed"]


def append_rows(rows, out_path, write_header):
    df = pd.DataFrame(rows)
    for c in FIXED_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[FIXED_COLS]
    df.to_csv(out_path, mode="a", header=write_header, index=False)
    return len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", choices=["eta_full", "eta_redundancy"], required=True)
    ap.add_argument("--seed-start", type=int, required=True)
    ap.add_argument("--seed-end", type=int, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()

    t0 = time.perf_counter()
    rows = []

    if args.sweep == "eta_full":
        for seed in range(args.seed_start, args.seed_end):
            for ds_name, cfg in DATASETS.items():
                ys, tau = cfg["gen"](seed)
                for eta in ETA_GRID:
                    row = run_full(ys, tau, eta, cfg["hazard"], cfg["pred_window"])
                    row.update(sweep="eta_full", dataset=ds_name, eta=eta,
                               method="Full CA-AOCP", seed=seed)
                    rows.append(row)
            print(f"seed {seed} done ({time.perf_counter()-t0:.1f}s)", flush=True)
    else:
        for seed in range(args.seed_start, args.seed_end):
            for ds_name, cfg in DATASETS.items():
                ys, tau = cfg["gen"](seed)
                for eta in ETA_GRID:
                    row_a = run_plain_aci(ys, tau, eta, cfg["pred_window"])
                    row_a.update(sweep="eta_redundancy", dataset=ds_name, eta=eta,
                                 method="ACI", seed=seed)
                    rows.append(row_a)
                    row_f = run_full(ys, tau, eta, cfg["hazard"], cfg["pred_window"])
                    row_f.update(sweep="eta_redundancy", dataset=ds_name, eta=eta,
                                 method="Full CA-AOCP", seed=seed)
                    rows.append(row_f)
            print(f"seed {seed} done ({time.perf_counter()-t0:.1f}s)", flush=True)

    n = append_rows(rows, out_path, write_header)
    print(f"Appended {n} rows to {out_path}")


if __name__ == "__main__":
    main()
