"""
Formal Significance Report: CA-AOCP vs 5 SOTA online-conformal baselines
==========================================================================
Re-analyzes run_baselines_multiseed.py's 30-seed x {Synthetic-CP,
Quality-Control} x 6-method output with PAIRED comparisons: for each
seed, all 6 methods (5 SOTA baselines + CA-AOCP) saw the EXACT SAME data
realization, so the correct test is on the paired difference per seed,
not a comparison of marginal means (same lesson as significance_report.py
for soft vs hard-reset).

For every (dataset x baseline x metric) cell, reports:
  - paired t-test:       mean diff, SE, t, p (two-sided)
  - Wilcoxon signed-rank: W, p (two-sided; robust at n=30)
  - BH-FDR corrected q-value (within each table)

Sign convention: diff is defined so that POSITIVE diff = CA-AOCP is
BETTER on that metric.
  - cov_gap / mean_width / winkler / recovery_step: lower is better,
    so diff = baseline_value - CA-AOCP_value.
  - postcov10 / postcov25 / postcov50: higher is better, so
    diff = CA-AOCP_value - baseline_value.

Usage:
    python3 experiments/significance_report_baselines.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "experiments" / "output_baselines_multiseed" / "raw.csv"
OUT_DIR = PROJECT_ROOT / "experiments" / "output_significance_baselines"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINES = ["SplitCP", "NExCP", "FACI", "SF-OGD", "SAOCP"]
LOWER_IS_BETTER = ["cov_gap", "mean_width", "winkler", "recovery_step"]
HIGHER_IS_BETTER = ["postcov10", "postcov25", "postcov50"]
METRICS = LOWER_IS_BETTER + HIGHER_IS_BETTER
DATASETS = ["Synthetic-CP", "Quality-Control"]


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    valid = ~np.isnan(p)
    q = np.full(n, np.nan)
    if valid.sum() == 0:
        return q
    idx = np.where(valid)[0]
    order = idx[np.argsort(p[idx])]
    ranked_p = p[order]
    m = len(order)
    q_sorted = ranked_p * m / (np.arange(m) + 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0, 1)
    q[order] = q_sorted
    return q


def paired_test(diff: np.ndarray) -> dict:
    diff = diff[~np.isnan(diff)]
    n = len(diff)
    if n < 3:
        return dict(n=n, mean_diff=np.nan, se=np.nan, t=np.nan, p_t=np.nan, W=np.nan, p_wilcoxon=np.nan)
    mean_diff = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(n)
    t_stat = mean_diff / se if se > 0 else np.nan
    p_t = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1)) if se > 0 else np.nan
    if np.allclose(diff, 0):
        W, p_w = np.nan, np.nan
    else:
        try:
            W, p_w = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            W, p_w = np.nan, np.nan
    return dict(n=n, mean_diff=mean_diff, se=se, t=t_stat, p_t=p_t, W=W, p_wilcoxon=p_w)


def sig_stars(p):
    if p is None or np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def main():
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded {len(df)} rows from {RAW_PATH}")
    print("Sanity check (should be 30 seeds x 2 datasets x 6 methods = 360):")
    print(df.groupby(["dataset", "Method"]).seed.nunique())

    rows = []
    for dataset in DATASETS:
        sub = df[df.dataset == dataset]
        wide = {}
        for metric in METRICS:
            piv = sub.pivot(index="seed", columns="Method", values=metric)
            wide[metric] = piv

        for baseline in BASELINES:
            for metric in METRICS:
                piv = wide[metric]
                if metric in LOWER_IS_BETTER:
                    diff = (piv[baseline] - piv["CA-AOCP"]).values
                else:
                    diff = (piv["CA-AOCP"] - piv[baseline]).values
                res = paired_test(diff)
                rows.append(dict(dataset=dataset, baseline=baseline, metric=metric, **res))

    out = pd.DataFrame(rows)
    out["q_t"] = np.nan
    out["q_wilcoxon"] = np.nan
    for dataset in DATASETS:
        m = out.dataset == dataset
        out.loc[m, "q_t"] = bh_fdr(out.loc[m, "p_t"].values)
        out.loc[m, "q_wilcoxon"] = bh_fdr(out.loc[m, "p_wilcoxon"].values)

    out["sig_t_fdr"] = out["q_t"].apply(sig_stars)
    out["sig_wilcoxon_fdr"] = out["q_wilcoxon"].apply(sig_stars)

    out.to_csv(OUT_DIR / "table_caaocp_vs_sota.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    for dataset in DATASETS:
        print(f"\n{'='*100}\n{dataset}: paired diff = CA-AOCP better direction (positive = CA-AOCP wins)\n{'='*100}")
        disp = out[out.dataset == dataset][
            ["baseline", "metric", "n", "mean_diff", "p_t", "q_t", "sig_t_fdr", "p_wilcoxon", "q_wilcoxon", "sig_wilcoxon_fdr"]
        ]
        print(disp.to_string(index=False))

    print(f"\nSaved to {OUT_DIR}/table_caaocp_vs_sota.csv")


if __name__ == "__main__":
    main()
