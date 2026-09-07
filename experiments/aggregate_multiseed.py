"""Aggregate the multi-seed ablation raw CSV into mean +/- SE / bootstrap CI tables."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IN_PATH = PROJECT_ROOT / "experiments" / "output_multiseed" / "raw.csv"
OUT_DIR = PROJECT_ROOT / "experiments" / "output_multiseed"

METRICS = ["coverage", "cov_gap", "mean_width", "winkler",
           "recovery_step", "postcov10", "postcov25", "postcov50",
           "false_alarm_rate", "cp_delay"]

METHOD_ORDER = ["ACI", "EWMA-ACI", "BOCPD-weighted CP", "BOCPD-weighted ACI",
                "CA-AOCP w/o ACI", "Full CA-AOCP", "Hard-reset CP", "Hard-reset ACI"]


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    df = pd.read_csv(IN_PATH)
    rows = []
    for ds in df["Dataset"].unique():
        sub_ds = df[df["Dataset"] == ds]
        for method in METHOD_ORDER:
            sub = sub_ds[sub_ds["Method"] == method]
            row = {"Dataset": ds, "Method": method, "n_seeds": sub["seed"].nunique()}
            for metric in METRICS:
                vals = sub[metric].values.astype(float)
                valid = vals[~np.isnan(vals)]
                mean = np.mean(valid) if len(valid) else np.nan
                se = np.std(valid, ddof=1) / np.sqrt(len(valid)) if len(valid) > 1 else np.nan
                lo, hi = bootstrap_ci(vals)
                row[f"{metric}_mean"] = mean
                row[f"{metric}_se"] = se
                row[f"{metric}_ci_lo"] = lo
                row[f"{metric}_ci_hi"] = hi
                row[f"{metric}_n"] = len(valid)
            rows.append(row)
    agg = pd.DataFrame(rows)
    agg.to_csv(OUT_DIR / "aggregate.csv", index=False)

    # Human-readable summary for the two headline metrics
    pd.set_option("display.width", 200)
    for ds in df["Dataset"].unique():
        print(f"\n{'='*90}\n{ds}  (n_seeds per method = {df[df.Dataset==ds]['seed'].nunique()})\n{'='*90}")
        sub = agg[agg["Dataset"] == ds].set_index("Method").loc[METHOD_ORDER]
        disp = pd.DataFrame({
            "cov_gap": [f"{m:.4f} ± {se:.4f}" for m, se in zip(sub.cov_gap_mean, sub.cov_gap_se)],
            "winkler": [f"{m:.3f} ± {se:.3f}" for m, se in zip(sub.winkler_mean, sub.winkler_se)],
            "postcov10": [f"{m:.3f} ± {se:.3f}" for m, se in zip(sub.postcov10_mean, sub.postcov10_se)],
            "postcov25": [f"{m:.3f} ± {se:.3f}" for m, se in zip(sub.postcov25_mean, sub.postcov25_se)],
            "recovery_step": [f"{m:.1f} ± {se:.1f}" if not np.isnan(m) else "NaN" for m, se in zip(sub.recovery_step_mean, sub.recovery_step_se)],
            "false_alarm": [f"{m:.4f}" if not np.isnan(m) else "-" for m in sub.false_alarm_rate_mean],
            "cp_delay": [f"{m:.2f}" if not np.isnan(m) else "NaN" for m in sub.cp_delay_mean],
        }, index=sub.index)
        print(disp.to_string())

    print(f"\nFull aggregate table saved to {OUT_DIR / 'aggregate.csv'}")


if __name__ == "__main__":
    main()
