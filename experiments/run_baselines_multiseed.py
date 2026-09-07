"""
Multi-seed SOTA baseline comparison for CA-AOCP significance testing.
=========================================================================
Extends run_baselines.py (which was single-seed, Synthetic-CP only) to:
  - N seeds x {Synthetic-CP, Quality-Control}
  - paired-by-seed design (every method sees the exact same data realization
    for a given seed/dataset), ready for significance_report-style paired
    testing
  - post-fix CAAOCP (max_run_length=50, matches paper Table 9 default;
    the O(K) truncation bug from earlier sections is already fixed in the
    ca_aocp package used here)

Baselines (from `online_conformal`, all genuinely published methods):
  SplitCP, NExCP, FACI, SF-OGD, SAOCP -- same 5 as run_baselines.py /
  run_real_data.py. None of them are changepoint-aware, so the metrics
  most likely to show a genuine CA-AOCP advantage are the post-change
  ones (postcov10/25/50, recovery_step), not necessarily whole-series
  cov_gap/width/winkler.

Usage:
    python3 experiments/run_baselines_multiseed.py --seed-start 0 --seed-end 3 \
        --out experiments/output_baselines_multiseed/raw.csv
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

from ca_aocp import CAAOCP
from ca_aocp.predictors import RollingMeanPredictor
from experiments.run_ablation import winkler_score
from experiments.run_multiseed import gen_synthetic_cp, gen_qc, recovery_step

from online_conformal.split_conformal import SplitConformal
from online_conformal.nex_conformal import NExConformal
from online_conformal.faci import FACI
from online_conformal.ogd import ScaleFreeOGD
from online_conformal.saocp import SAOCP


# ═══════════════════════════════════════════════════════════════════════
# Configuration -- matches run_multiseed.py / run_baselines.py defaults
# ═══════════════════════════════════════════════════════════════════════

COVERAGE = 0.9
ALPHA = 1.0 - COVERAGE
MAX_SCALE = 10.0  # expected max |y - yhat|, used by SF-OGD / SAOCP init

DATASETS = {
    "Synthetic-CP": dict(gen=gen_synthetic_cp, hazard=0.005, pred_window=20),
    "Quality-Control": dict(gen=gen_qc, hazard=0.010, pred_window=15),
}

CAAOCP_COMMON = dict(
    alpha=ALPHA,
    eta=0.02,
    decay=0.01,
    temperature=0.5,
    bocpd_mu0=0.0,
    bocpd_kappa0=1.0,
    bocpd_alpha0=2.0,
    bocpd_beta0=1.0,
    max_run_length=50,   # post-fix: paper Table 9 default (was 400 in the stale run)
    init_radius=1.0,
)


def make_predictor(pred_window: int) -> RollingMeanPredictor:
    return RollingMeanPredictor(window=pred_window)


# ═══════════════════════════════════════════════════════════════════════
# One (seed, dataset) run: 5 SOTA baselines + CA-AOCP
# ═══════════════════════════════════════════════════════════════════════

def run_one(ys: np.ndarray, tau: int, hazard: float, pred_window: int) -> list[dict]:
    methods: dict[str, object] = {}
    methods["SplitCP"] = SplitConformal(None, None, coverage=COVERAGE)
    methods["NExCP"] = NExConformal(None, None, coverage=COVERAGE)
    methods["FACI"] = FACI(None, None, coverage=COVERAGE)
    methods["SF-OGD"] = ScaleFreeOGD(None, None, coverage=COVERAGE, max_scale=MAX_SCALE)
    methods["SAOCP"] = SAOCP(None, None, coverage=COVERAGE, max_scale=MAX_SCALE)

    caaocp = CAAOCP(predictor=make_predictor(pred_window), hazard=hazard, **CAAOCP_COMMON)
    methods["CA-AOCP"] = caaocp

    shared_pred = make_predictor(pred_window)
    names = list(methods.keys())
    buffers = {n: {"covered": [], "width": [], "lower": [], "upper": []} for n in names}

    T = len(ys)
    for t_idx in range(T):
        y_t = float(ys[t_idx])
        yhat_shared = shared_pred.predict(None)

        for name in names:
            method = methods[name]
            if name == "CA-AOCP":
                lo, hi = method.predict(None)
                method.update(None, y_t)
            else:
                delta_lb, delta_ub = method.predict(horizon=1)
                lo = yhat_shared + delta_lb
                hi = yhat_shared + delta_ub
                method.update(
                    ground_truth=pd.Series([y_t]),
                    forecast=pd.Series([yhat_shared]),
                    horizon=1,
                )
            covered = int(lo <= y_t <= hi)
            buffers[name]["covered"].append(covered)
            buffers[name]["width"].append(hi - lo)
            buffers[name]["lower"].append(lo)
            buffers[name]["upper"].append(hi)

        shared_pred.update(None, y_t)

    rows = []
    for name in names:
        cov = np.array(buffers[name]["covered"], dtype=float)
        lo = np.array(buffers[name]["lower"])
        hi = np.array(buffers[name]["upper"])
        width = np.array(buffers[name]["width"])
        w = winkler_score(ys, lo, hi, ALPHA)
        row = {
            "Method": name,
            "coverage": cov.mean(),
            "cov_gap": abs(cov.mean() - COVERAGE),
            "mean_width": width.mean(),
            "winkler": w.mean(),
            "recovery_step": recovery_step(cov.astype(int), tau, ALPHA),
        }
        # postcov_h: raw post-change coverage in the first h steps after tau
        #   (useful as a "how fast does coverage come back up" race metric
        #   when baselines are badly undercovering right after the shift --
        #   but NOT a calibration-accuracy metric on its own, since coverage
        #   above target is not "better" -- see postgap_h below).
        # postgap_h: |postcov_h - target|, the calibration-accuracy analogue
        #   of cov_gap but restricted to the post-change window. Lower is
        #   better. This is what should be used once methods have mostly
        #   recovered (larger h), where "higher postcov" can just mean
        #   "more overcoverage", not "better calibrated".
        # post_winkler_h / post_width_h: Winkler score / mean interval width
        #   restricted to the same post-change window -- lets us check
        #   whether a post-change coverage/postgap advantage survives once
        #   interval width is accounted for (rather than just reflecting
        #   systematically wider intervals).
        for h in (10, 25, 50):
            end = min(tau + h, T)
            seg_cov = cov[tau:end]
            seg_w = w[tau:end]
            seg_width = width[tau:end]
            if len(seg_cov) > 0:
                postcov_h = seg_cov.mean()
                row[f"postcov{h}"] = postcov_h
                row[f"postgap{h}"] = abs(postcov_h - COVERAGE)
                row[f"post_winkler{h}"] = seg_w.mean()
                row[f"post_width{h}"] = seg_width.mean()
            else:
                row[f"postcov{h}"] = np.nan
                row[f"postgap{h}"] = np.nan
                row[f"post_winkler{h}"] = np.nan
                row[f"post_width{h}"] = np.nan
        rows.append(row)
    return rows


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-start", type=int, required=True)
    ap.add_argument("--seed-end", type=int, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()

    all_rows = []
    t0 = time.perf_counter()
    for seed in range(args.seed_start, args.seed_end):
        for ds_name, cfg in DATASETS.items():
            ys, tau = cfg["gen"](seed)
            rows = run_one(ys, tau, cfg["hazard"], cfg["pred_window"])
            for r in rows:
                r["dataset"] = ds_name
                r["seed"] = seed
            all_rows.extend(rows)
        print(f"seed {seed} done ({time.perf_counter()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(all_rows)
    fixed_cols = ["dataset", "seed", "Method", "coverage", "cov_gap", "mean_width",
                  "winkler", "recovery_step",
                  "postcov10", "postgap10", "post_winkler10", "post_width10",
                  "postcov25", "postgap25", "post_winkler25", "post_width25",
                  "postcov50", "postgap50", "post_winkler50", "post_width50"]
    df = df[fixed_cols]
    df.to_csv(out_path, mode="a", header=write_header, index=False)
    print(f"Appended {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
