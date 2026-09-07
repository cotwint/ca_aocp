"""
Formal Significance Report: soft (BOCPD-weighted) vs hard-reset
====================================================================
Re-analyzes BOTH prior experiments (run_multiseed.py's 30-seed ablation,
run_shift_grid.py's 21-seed x 10-condition grid) with the statistically
correct PAIRED comparison: for each seed, hard-reset and soft weighting
saw the EXACT SAME data realization, so the right test is on the paired
difference (hard - soft) per seed, not a comparison of marginal means.

For every metric x pairing (CP-variant / ACI-variant) x dataset-or-condition
cell, reports:
  - paired t-test:      mean diff, SE, t, p (two-sided)
  - Wilcoxon signed-rank: W, p (two-sided; robust to non-normal diffs,
                           more trustworthy at n=21-30)
  - BH-FDR corrected q-value (within each table, since many tests are run)

Positive mean_diff = hard-reset is WORSE (since diff = hard - soft, and for
cov_gap/mean_width/winkler/width_overshoot lower-is-better; for
coverage/postcov*/neff* higher mean_diff = hard-reset HIGHER on that metric
-- direction is metric-specific, see the `higher_is_worse_for_hard` note
printed per table).

Usage:
    python3 experiments/significance_report.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MULTISEED_PATH = PROJECT_ROOT / "experiments" / "output_multiseed" / "raw.csv"
SHIFTGRID_PATH = PROJECT_ROOT / "experiments" / "output_shiftgrid" / "raw.csv"
OUT_DIR = PROJECT_ROOT / "experiments" / "output_significance"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns q-values (same order as input)."""
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
    # enforce monotonicity from the largest p-value down
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
    t_res = stats.ttest_rel(diff, np.zeros_like(diff)) if False else None
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
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ═══════════════════════════════════════════════════════════════════════
# Table A: multi-seed ablation (Synthetic-CP / Quality-Control), 30 seeds
# ═══════════════════════════════════════════════════════════════════════

MULTISEED_METRICS = ["coverage", "cov_gap", "mean_width", "winkler",
                      "recovery_step", "postcov10", "postcov25", "postcov50"]
PAIRINGS = [("BOCPD-weighted CP", "Hard-reset CP", "CP (no ACI)"),
            ("BOCPD-weighted ACI", "Hard-reset ACI", "ACI")]


def build_table_A():
    df = pd.read_csv(MULTISEED_PATH)
    rows = []
    for ds in df["Dataset"].unique():
        sub_ds = df[df["Dataset"] == ds]
        for soft_m, hard_m, pairing_label in PAIRINGS:
            piv = sub_ds.pivot_table(index="seed", columns="Method", values=MULTISEED_METRICS)
            for metric in MULTISEED_METRICS:
                d = (piv[(metric, hard_m)] - piv[(metric, soft_m)]).values
                res = paired_test(d)
                rows.append(dict(experiment="multiseed_ablation", dataset=ds, pairing=pairing_label,
                                  metric=metric, **res))
    out = pd.DataFrame(rows)
    out["q_wilcoxon"] = bh_fdr(out["p_wilcoxon"].values)
    out["sig"] = out["p_wilcoxon"].apply(sig_stars)
    return out


# ═══════════════════════════════════════════════════════════════════════
# Table B: shift grid, pooled across all 10 conditions (210 paired obs)
# ═══════════════════════════════════════════════════════════════════════

SHIFTGRID_METRICS = ["cov_gap", "mean_width", "postcov10", "postcov25", "postcov50",
                      "postcov100", "width_overshoot_ratio",
                      "neff_t+1", "neff_t+5", "neff_t+10", "neff_t+25"]


def build_table_B():
    df = pd.read_csv(SHIFTGRID_PATH)
    rows = []
    for soft_m, hard_m, pairing_label in PAIRINGS:
        piv = df.pivot_table(index=["condition", "seed"], columns="Method", values=SHIFTGRID_METRICS)
        for metric in SHIFTGRID_METRICS:
            d = (piv[(metric, hard_m)] - piv[(metric, soft_m)]).values
            res = paired_test(d)
            rows.append(dict(experiment="shift_grid_pooled", dataset="ALL_10_CONDITIONS",
                              pairing=pairing_label, metric=metric, **res))
    out = pd.DataFrame(rows)
    out["q_wilcoxon"] = bh_fdr(out["p_wilcoxon"].values)
    out["sig"] = out["p_wilcoxon"].apply(sig_stars)
    return out


# ═══════════════════════════════════════════════════════════════════════
# Table C: shift grid, per-condition breakdown for the 4 headline metrics
# ═══════════════════════════════════════════════════════════════════════

HEADLINE_METRICS = ["cov_gap", "mean_width", "postcov10", "width_overshoot_ratio"]


def build_table_C():
    df = pd.read_csv(SHIFTGRID_PATH)
    rows = []
    for cond in df["condition"].unique():
        sub = df[df["condition"] == cond]
        for soft_m, hard_m, pairing_label in PAIRINGS:
            piv = sub.pivot_table(index="seed", columns="Method", values=HEADLINE_METRICS)
            for metric in HEADLINE_METRICS:
                d = (piv[(metric, hard_m)] - piv[(metric, soft_m)]).values
                res = paired_test(d)
                rows.append(dict(experiment="shift_grid_by_condition", dataset=cond,
                                  pairing=pairing_label, metric=metric, **res))
    out = pd.DataFrame(rows)
    out["q_wilcoxon"] = bh_fdr(out["p_wilcoxon"].values)
    out["sig"] = out["p_wilcoxon"].apply(sig_stars)
    return out


def main():
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 400)

    A = build_table_A()
    B = build_table_B()
    C = build_table_C()

    A.to_csv(OUT_DIR / "table_A_multiseed_ablation.csv", index=False)
    B.to_csv(OUT_DIR / "table_B_shiftgrid_pooled.csv", index=False)
    C.to_csv(OUT_DIR / "table_C_shiftgrid_by_condition.csv", index=False)

    fmt_cols = ["experiment", "dataset", "pairing", "metric", "n", "mean_diff", "se", "t",
                "p_t", "p_wilcoxon", "q_wilcoxon", "sig"]

    print("\n" + "=" * 130)
    print("TABLE A: multi-seed ablation (n=30 seeds), paired diff = Hard-reset - BOCPD-weighted")
    print("=" * 130)
    print(A[fmt_cols].round(4).to_string(index=False))

    print("\n" + "=" * 130)
    print("TABLE B: shift grid POOLED across all 10 conditions (n=210 paired obs)")
    print("=" * 130)
    print(B[fmt_cols].round(4).to_string(index=False))

    print("\n" + "=" * 130)
    print("TABLE C: shift grid BY CONDITION, headline metrics only (n=21 seeds each)")
    print("=" * 130)
    print(C[fmt_cols].round(4).to_string(index=False))

    # Summary: which findings survive FDR correction at q<0.05
    print("\n" + "=" * 130)
    print("SURVIVING FDR CORRECTION (q_wilcoxon < 0.05) -- across all three tables combined")
    print("=" * 130)
    combined = pd.concat([A, B, C], ignore_index=True)
    combined["q_wilcoxon_global"] = bh_fdr(combined["p_wilcoxon"].values)
    surviving = combined[combined["q_wilcoxon_global"] < 0.05].sort_values("q_wilcoxon_global")
    print(surviving[["experiment", "dataset", "pairing", "metric", "n", "mean_diff", "p_wilcoxon", "q_wilcoxon_global"]]
          .round(4).to_string(index=False))
    combined.to_csv(OUT_DIR / "table_combined_with_global_FDR.csv", index=False)

    print(f"\nSaved: {OUT_DIR}/table_A_multiseed_ablation.csv")
    print(f"Saved: {OUT_DIR}/table_B_shiftgrid_pooled.csv")
    print(f"Saved: {OUT_DIR}/table_C_shiftgrid_by_condition.csv")
    print(f"Saved: {OUT_DIR}/table_combined_with_global_FDR.csv")


if __name__ == "__main__":
    main()
