"""Aggregate the shift-grid raw CSV: mean/SE for point metrics, plus explicit
cross-seed STD for the robustness/volatility metrics (width_overshoot_ratio,
neff_t+1/5) which is the whole point of this experiment -- soft vs hard
weighting is hypothesised to differ in *volatility*, not necessarily in
mean coverage.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IN_PATH = PROJECT_ROOT / "experiments" / "output_shiftgrid" / "raw.csv"
OUT_DIR = PROJECT_ROOT / "experiments" / "output_shiftgrid"

METHOD_ORDER = ["BOCPD-weighted CP", "Hard-reset CP", "BOCPD-weighted ACI", "Hard-reset ACI"]


def main():
    df = pd.read_csv(IN_PATH)
    conditions = list(df["condition"].unique())

    metrics_mean = ["cov_gap", "postcov10", "postcov25", "width_overshoot_ratio",
                     "neff_t+1", "neff_t+5", "neff_t+10"]
    rows = []
    for cond in conditions:
        for method in METHOD_ORDER:
            sub = df[(df.condition == cond) & (df.Method == method)]
            row = {"condition": cond, "Method": method, "n": len(sub)}
            for m in metrics_mean:
                vals = sub[m].values.astype(float)
                valid = vals[~np.isnan(vals)]
                row[f"{m}_mean"] = np.mean(valid) if len(valid) else np.nan
                row[f"{m}_std"] = np.std(valid, ddof=1) if len(valid) > 1 else np.nan
            rows.append(row)
    agg = pd.DataFrame(rows)
    agg.to_csv(OUT_DIR / "aggregate.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 100)
    print("HEADLINE: cov_gap (mean ± std across seeds) and width_overshoot_ratio (mean ± std)")
    print("=" * 100)
    for cond in conditions:
        sub = agg[agg.condition == cond].set_index("Method").loc[METHOD_ORDER]
        print(f"\n--- {cond} ---")
        disp = pd.DataFrame({
            "cov_gap": [f"{m:.4f} ± {s:.4f}" for m, s in zip(sub.cov_gap_mean, sub.cov_gap_std)],
            "postcov10": [f"{m:.3f} ± {s:.3f}" for m, s in zip(sub.postcov10_mean, sub.postcov10_std)],
            "overshoot": [f"{m:.3f} ± {s:.3f}" for m, s in zip(sub.width_overshoot_ratio_mean, sub.width_overshoot_ratio_std)],
            "Neff@t+1": [f"{m:.1f} ± {s:.1f}" for m, s in zip(sub["neff_t+1_mean"], sub["neff_t+1_std"])],
            "Neff@t+5": [f"{m:.1f} ± {s:.1f}" for m, s in zip(sub["neff_t+5_mean"], sub["neff_t+5_std"])],
        }, index=sub.index)
        print(disp.to_string())

    # Direct soft-vs-hard paired comparison per condition (CP variant, no-ACI)
    print("\n" + "=" * 100)
    print("SOFT vs HARD gap in overshoot volatility (std), CP variant (no ACI)")
    print("std(width_overshoot_ratio): higher = more erratic across seeds")
    print("=" * 100)
    for cond in conditions:
        soft = agg[(agg.condition == cond) & (agg.Method == "BOCPD-weighted CP")].iloc[0]
        hard = agg[(agg.condition == cond) & (agg.Method == "Hard-reset CP")].iloc[0]
        print(f"{cond:28s}  soft_std={soft.width_overshoot_ratio_std:.3f}  hard_std={hard.width_overshoot_ratio_std:.3f}  "
              f"soft_Neff@1={soft['neff_t+1_mean']:.1f}  hard_Neff@1={hard['neff_t+1_mean']:.1f}")

    print(f"\nFull table saved to {OUT_DIR / 'aggregate.csv'}")


if __name__ == "__main__":
    main()
