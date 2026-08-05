"""Aggregate the eta sensitivity + redundancy experiment.

(A) eta_full: mean +/- SE of Full CA-AOCP's metrics at each eta, per dataset,
    plus %-change vs the paper's default eta=0.02 -- to see whether the
    cov_gap penalty documented in Task #5 (hyperparameter sensitivity)
    shrinks as eta shrinks.

(B) eta_redundancy: paired difference (Full CA-AOCP - plain ACI) per seed,
    per eta, per dataset, with a paired t-test -- to see whether there is
    ANY eta at which Full CA-AOCP's own BOCPD+decay machinery earns its
    keep over plain ACI on the same data draw.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IN_PATH = PROJECT_ROOT / "experiments" / "output_eta" / "raw.csv"
OUT_DIR = PROJECT_ROOT / "experiments" / "output_eta"

METRICS = ["cov_gap", "mean_width", "winkler", "postcov10", "postcov25", "recovery_step"]
DEFAULT_ETA = 0.02


def paired_test(diff):
    diff = diff[~np.isnan(diff)]
    n = len(diff)
    if n < 3:
        return dict(n=n, mean_diff=np.nan, se=np.nan, t=np.nan, p=np.nan)
    mean_diff = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(n)
    t_stat = mean_diff / se if se > 0 else np.nan
    p = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1)) if se > 0 else np.nan
    return dict(n=n, mean_diff=mean_diff, se=se, t=t_stat, p=p)


def main():
    df = pd.read_csv(IN_PATH)
    pd.set_option("display.width", 160)

    # ---------------- (A) eta_full ----------------
    full = df[df.sweep == "eta_full"]
    rows = []
    for ds in full.dataset.unique():
        sub = full[full.dataset == ds]
        base = sub[sub.eta == DEFAULT_ETA][METRICS].mean()
        for eta, g in sub.groupby("eta"):
            row = {"dataset": ds, "eta": eta}
            for m in METRICS:
                vals = g[m].values.astype(float)
                valid = vals[~np.isnan(vals)]
                mean = np.mean(valid) if len(valid) else np.nan
                se = np.std(valid, ddof=1) / np.sqrt(len(valid)) if len(valid) > 1 else np.nan
                pct = 100 * (mean - base[m]) / base[m] if base[m] != 0 else np.nan
                row[f"{m}_mean"] = mean
                row[f"{m}_se"] = se
                row[f"{m}_pct_vs_default"] = pct
            rows.append(row)
    out_a = pd.DataFrame(rows).sort_values(["dataset", "eta"])
    out_a.to_csv(OUT_DIR / "sensitivity_eta_full.csv", index=False)

    print("=" * 110)
    print("(A) eta_full: Full CA-AOCP sensitivity to eta (default=0.02, n=10 seeds/point)")
    print("=" * 110)
    for ds in out_a.dataset.unique():
        sub = out_a[out_a.dataset == ds]
        disp = pd.DataFrame({
            "cov_gap": [f"{m:.4f}±{s:.4f} ({p:+.0f}%)" for m, s, p in zip(sub.cov_gap_mean, sub.cov_gap_se, sub.cov_gap_pct_vs_default)],
            "mean_width": [f"{m:.3f}±{s:.3f} ({p:+.0f}%)" for m, s, p in zip(sub.mean_width_mean, sub.mean_width_se, sub.mean_width_pct_vs_default)],
            "winkler": [f"{m:.3f}±{s:.3f} ({p:+.0f}%)" for m, s, p in zip(sub.winkler_mean, sub.winkler_se, sub.winkler_pct_vs_default)],
            "postcov10": [f"{m:.3f}±{s:.3f}" for m, s in zip(sub.postcov10_mean, sub.postcov10_se)],
        }, index=sub.eta)
        print(f"\n-- {ds} --")
        print(disp.to_string())

    # ---------------- (B) eta_redundancy ----------------
    red = df[df.sweep == "eta_redundancy"]
    rows = []
    for ds in red.dataset.unique():
        sub_ds = red[red.dataset == ds]
        for eta, g in sub_ds.groupby("eta"):
            piv = g.pivot_table(index="seed", columns="method", values=METRICS)
            row_base = {"dataset": ds, "eta": eta}
            for m in METRICS:
                d = (piv[(m, "Full CA-AOCP")] - piv[(m, "ACI")]).values
                res = paired_test(d)
                for k, v in res.items():
                    row_base[f"{m}_{k}"] = v
            rows.append(row_base)
    out_b = pd.DataFrame(rows).sort_values(["dataset", "eta"])
    out_b.to_csv(OUT_DIR / "redundancy_eta.csv", index=False)

    print("\n" + "=" * 110)
    print("(B) eta_redundancy: paired diff (Full CA-AOCP - plain ACI), n=10 seeds/eta")
    print("positive cov_gap/winkler/mean_width diff = Full CA-AOCP WORSE than plain ACI")
    print("=" * 110)
    for ds in out_b.dataset.unique():
        sub = out_b[out_b.dataset == ds]
        disp = pd.DataFrame({
            "cov_gap_diff": [f"{m:+.4f} (p={p:.3f})" for m, p in zip(sub.cov_gap_mean_diff, sub.cov_gap_p)],
            "winkler_diff": [f"{m:+.3f} (p={p:.3f})" for m, p in zip(sub.winkler_mean_diff, sub.winkler_p)],
            "mean_width_diff": [f"{m:+.3f} (p={p:.3f})" for m, p in zip(sub.mean_width_mean_diff, sub.mean_width_p)],
            "postcov10_diff": [f"{m:+.3f} (p={p:.3f})" for m, p in zip(sub.postcov10_mean_diff, sub.postcov10_p)],
        }, index=sub.eta)
        print(f"\n-- {ds} --")
        print(disp.to_string())

    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
