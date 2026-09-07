"""Aggregate hyperparameter sensitivity sweep: mean +/- SE per swept value,
plus %-change relative to the default point, so "how sensitive" has a
concrete number attached to it (not just eyeballing a table)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IN_PATH = PROJECT_ROOT / "experiments" / "output_hparam" / "raw.csv"
OUT_DIR = PROJECT_ROOT / "experiments" / "output_hparam"

DEFAULTS = dict(hazard=0.005, decay=0.01, K=50)
METRICS = ["cov_gap", "mean_width", "winkler", "postcov10", "postcov25", "recovery_step"]


def main():
    df = pd.read_csv(IN_PATH)
    pd.set_option("display.width", 160)

    for sweep_name in ["hazard", "decay", "K"]:
        sub = df[df.sweep == sweep_name]
        default_val = DEFAULTS[sweep_name]
        base = sub[sub.swept_value == default_val][METRICS].mean()

        rows = []
        for v, g in sub.groupby("swept_value"):
            row = {"swept_value": v}
            for m in METRICS:
                vals = g[m].values.astype(float)
                valid = vals[~np.isnan(vals)]
                mean = np.mean(valid) if len(valid) else np.nan
                se = np.std(valid, ddof=1) / np.sqrt(len(valid)) if len(valid) > 1 else np.nan
                pct_change = 100 * (mean - base[m]) / base[m] if base[m] != 0 else np.nan
                row[f"{m}_mean"] = mean
                row[f"{m}_se"] = se
                row[f"{m}_pct_vs_default"] = pct_change
            rows.append(row)
        out = pd.DataFrame(rows).sort_values("swept_value")
        out.to_csv(OUT_DIR / f"sensitivity_{sweep_name}.csv", index=False)

        print(f"\n{'='*100}\nSweep: {sweep_name}  (default={default_val}, n=10 seeds/point)\n{'='*100}")
        disp = pd.DataFrame({
            "cov_gap": [f"{m:.4f}±{s:.4f} ({p:+.0f}%)" for m, s, p in zip(out.cov_gap_mean, out.cov_gap_se, out.cov_gap_pct_vs_default)],
            "mean_width": [f"{m:.3f}±{s:.3f} ({p:+.0f}%)" for m, s, p in zip(out.mean_width_mean, out.mean_width_se, out.mean_width_pct_vs_default)],
            "winkler": [f"{m:.3f}±{s:.3f} ({p:+.0f}%)" for m, s, p in zip(out.winkler_mean, out.winkler_se, out.winkler_pct_vs_default)],
            "postcov10": [f"{m:.3f}±{s:.3f}" for m, s in zip(out.postcov10_mean, out.postcov10_se)],
            "recovery_step": [f"{m:.2f}±{s:.2f}" for m, s in zip(out.recovery_step_mean, out.recovery_step_se)],
        }, index=out.swept_value)
        print(disp.to_string())

    # cross_hazard
    print(f"\n{'='*100}\ncross_hazard: matched vs mismatched hazard, per dataset\n{'='*100}")
    ch = df[df.sweep == "cross_hazard"].copy()
    print("sanity check (should show 2 tags x 2 datasets, n=10 seeds each):")
    print(ch.groupby(["dataset", "tag"]).seed.nunique())
    rows = []
    for (ds, tag), g in ch.groupby(["dataset", "tag"]):
        row = {"dataset": ds, "tag": tag}
        for m in METRICS:
            vals = g[m].values.astype(float)
            valid = vals[~np.isnan(vals)]
            row[f"{m}_mean"] = np.mean(valid) if len(valid) else np.nan
            row[f"{m}_se"] = np.std(valid, ddof=1) / np.sqrt(len(valid)) if len(valid) > 1 else np.nan
        rows.append(row)
    ch_out = pd.DataFrame(rows)
    ch_out.to_csv(OUT_DIR / "sensitivity_cross_hazard.csv", index=False)
    disp2 = pd.DataFrame({
        "cov_gap": [f"{m:.4f}±{s:.4f}" for m, s in zip(ch_out.cov_gap_mean, ch_out.cov_gap_se)],
        "mean_width": [f"{m:.3f}±{s:.3f}" for m, s in zip(ch_out.mean_width_mean, ch_out.mean_width_se)],
        "winkler": [f"{m:.3f}±{s:.3f}" for m, s in zip(ch_out.winkler_mean, ch_out.winkler_se)],
        "postcov10": [f"{m:.3f}±{s:.3f}" for m, s in zip(ch_out.postcov10_mean, ch_out.postcov10_se)],
    }, index=[ch_out.dataset, ch_out.tag])
    print(disp2.to_string())

    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
