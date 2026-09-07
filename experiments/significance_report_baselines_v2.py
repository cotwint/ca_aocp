"""
Formal Significance Report v2: CA-AOCP vs 5 SOTA online-conformal baselines
==============================================================================
Revision of significance_report_baselines.py after review feedback on
section 22 of the experiment summary. Two fixes:

1. Uses output_baselines_multiseed_v2/raw.csv, which adds postgap_h,
   post_winkler_h, post_width_h (h in 10/25/50) alongside the original
   postcov_h -- see run_baselines_multiseed.py for the additions. The
   underlying simulation is byte-identical to the v1 run (verified: all
   original columns match to 0.0 abs diff), only more diagnostics are
   extracted per run.

2. `postcov_h` alone is not a calibration-accuracy metric once methods
   have mostly recovered: coverage above target is not "better" than
   coverage at target. Treating raw postcov_h as monotonically
   "higher is better" is only really justified at short horizons (h=10)
   where baselines are still badly *undercovering* -- at h=25/50 it can
   reward pure overcoverage. `postgap_h = |postcov_h - target|` is the
   correct calibration-accuracy analogue of cov_gap restricted to the
   post-change window, and is reported alongside postcov_h here (both are
   kept -- postcov_h answers "who gets back to nominal coverage fastest",
   postgap_h answers "who is best-calibrated in the post-change window").

Also reports post_winkler_h / post_width_h (lower is better) to check
whether a post-change coverage/postgap advantage survives once interval
width is accounted for, since CA-AOCP's intervals are systematically
wider than SplitCP/NExCP/FACI (see section 22 discussion) -- a pure
width confound could otherwise explain apparent "faster recovery".

Sign convention unchanged: diff is defined so POSITIVE = CA-AOCP better.
  - cov_gap / mean_width / winkler / recovery_step / postgap_h /
    post_winkler_h / post_width_h: lower is better,
    diff = baseline_value - CA-AOCP_value.
  - postcov10 / postcov25 / postcov50: higher is better,
    diff = CA-AOCP_value - baseline_value.

Usage:
    python3 experiments/significance_report_baselines_v2.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "experiments" / "output_baselines_multiseed_v2" / "raw.csv"
OUT_DIR = PROJECT_ROOT / "experiments" / "output_significance_baselines_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINES = ["SplitCP", "NExCP", "FACI", "SF-OGD", "SAOCP"]
LOWER_IS_BETTER = [
    "cov_gap", "mean_width", "winkler", "recovery_step",
    "postgap10", "postgap25", "postgap50",
    "post_winkler10", "post_winkler25", "post_winkler50",
    "post_width10", "post_width25", "post_width50",
]
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


def overcoverage_sign_test(cov: np.ndarray, target: float) -> dict:
    """Binomial sign test: is coverage systematically above target across seeds?"""
    cov = cov[~np.isnan(cov)]
    n = len(cov)
    n_over = int((cov > target).sum())
    n_under = int((cov < target).sum())
    n_eq = n - n_over - n_under
    # two-sided binomial test on n_over vs n (excluding ties), H0: p=0.5
    n_eff = n_over + n_under
    if n_eff == 0:
        p = np.nan
    else:
        res = stats.binomtest(n_over, n_eff, 0.5, alternative="two-sided")
        p = res.pvalue
    return dict(n=n, n_over=n_over, n_under=n_under, n_eq=n_eq, p_sign=p)


def main():
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded {len(df)} rows from {RAW_PATH}")

    # ── Overcoverage diagnostic for CA-AOCP itself (not a baseline comparison) ──
    print(f"\n{'='*90}\nCA-AOCP systematic overcoverage check (target = 0.9)\n{'='*90}")
    for dataset in DATASETS:
        cov = df[(df.dataset == dataset) & (df.Method == "CA-AOCP")]["coverage"].values
        res = overcoverage_sign_test(cov, 0.9)
        mean_cov = np.nanmean(cov)
        print(f"{dataset:16s}: mean coverage={mean_cov:.4f}  seeds over target={res['n_over']}/{res['n']}  "
              f"under={res['n_under']}/{res['n']}  sign-test p={res['p_sign']:.2e}")

    # ── Paired comparisons vs each baseline ──
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

    out.to_csv(OUT_DIR / "table_caaocp_vs_sota_v2.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 300)
    for dataset in DATASETS:
        print(f"\n{'='*100}\n{dataset}: paired diff, positive = CA-AOCP better\n{'='*100}")
        disp = out[out.dataset == dataset][
            ["baseline", "metric", "n", "mean_diff", "p_wilcoxon", "q_wilcoxon", "sig_wilcoxon_fdr"]
        ]
        print(disp.to_string(index=False))

    print(f"\nSaved to {OUT_DIR}/table_caaocp_vs_sota_v2.csv")


if __name__ == "__main__":
    main()
