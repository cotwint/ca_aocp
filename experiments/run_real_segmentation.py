"""
Real Data With GROUND-TRUTH Discrete Changepoints
=====================================================
The previous real-data attempt (run_real_multiregime.py: MSFT volatility,
Taylor electricity demand) showed that CA-AOCP's BOCPD-based methods do
*not* have an advantage on data with continuous/gradual regime evolution or
strong deterministic seasonality the predictor can't track -- both are
qualitatively different from the paper's own synthetic benchmark design
(a single, discrete, human-labeled mean-shift changepoint).

This script instead uses real sensor time series from the aeon time-series
*segmentation* benchmark (Ermshaus et al.), which are built by concatenating
genuinely different real operating regimes and recording the TRUE splice
points as ground-truth change points -- i.e. real data, but with the same
"discrete regime switch" structure the paper's theory and synthetic
experiments assume. Both are bundled offline with the `aeon` package (no
network fetch needed in this sandbox):

  - electric_devices_segmentation: real UCR "Electric Devices" power-usage
    traces concatenated across device classes; 4 labeled change points.
    We carve out 3 non-overlapping +/-600-step windows around 3 of them
    (skipping the 4th to avoid edge effects) as 3 separate real episodes.
  - gun_point_segmentation: real motion-capture (gun-draw) sensor data,
    1 labeled change point (a genuine "sensor mode switch", closer to a
    process/fault-monitoring signal than to financial or seasonal data).

Usage:
    python3 experiments/run_real_segmentation.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_ablation import make_methods, winkler_score


def summarize(name, res, ys, alpha, tau):
    cov = np.asarray(res["covered"], dtype=float)
    lo = np.asarray(res["lower"]); hi = np.asarray(res["upper"])
    width = np.asarray(res["width"])
    w = winkler_score(np.asarray(ys), lo, hi, alpha)
    T = len(ys)

    row = {
        "Method": name,
        "coverage": cov.mean(),
        "cov_gap": abs(cov.mean() - (1 - alpha)),
        "mean_width": width.mean(),
        "winkler": w.mean(),
    }
    for h in (10, 25, 50, 100):
        end = min(tau + h, T)
        seg = cov[tau:end]
        row[f"postcov{h}"] = seg.mean() if len(seg) else np.nan
    return row


def run_episode(ys, tau, alpha, eta, temperature, decay, hazard, bocpd_kwargs, K,
                 pred_window, init_radius, alpha_min, alpha_max):
    methods = make_methods(alpha, eta, temperature, decay, hazard, bocpd_kwargs, K,
                            pred_window, init_radius, alpha_min, alpha_max)
    rows = []
    for mname, m in methods.items():
        res = m.run(ys)
        rows.append(summarize(mname, res, ys, alpha, tau))
    return pd.DataFrame(rows).set_index("Method")


def build_electric_devices_episodes(window_radius: int = 600):
    from aeon.datasets import load_electric_devices_segmentation
    X, period_length, change_points = load_electric_devices_segmentation()
    y_full = X.values.astype(float)
    episodes = []
    usable_cps = [cp for cp in change_points if cp - window_radius >= 0 and cp + window_radius <= len(y_full)]
    # keep non-overlapping ones only
    kept = []
    for cp in usable_cps:
        if all(abs(cp - k) > 2 * window_radius for k in kept):
            kept.append(cp)
    for cp in kept[:3]:
        start, end = cp - window_radius, cp + window_radius
        ys = y_full[start:end]
        tau = window_radius
        episodes.append((f"ElectricDevices_cp{cp}", ys, tau))
    return episodes


def build_gunpoint_episode():
    from aeon.datasets import load_gun_point_segmentation
    X, period_length, change_points = load_gun_point_segmentation()
    ys = X.values.astype(float)
    tau = int(change_points[0])
    return [("GunPoint_sensor", ys, tau)]


def main():
    alpha = 0.1
    eta = 0.02
    temperature = 0.5
    decay = 0.01
    bocpd_kwargs = dict(mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=1.0)
    K = 50
    alpha_min, alpha_max = alpha / 2, (1 + alpha) / 2

    out_dir = PROJECT_ROOT / "experiments" / "output_real_segmentation"
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.set_option("display.width", 180)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    episodes = build_electric_devices_episodes() + build_gunpoint_episode()

    for name, ys, tau in episodes:
        init_radius = float(np.std(ys[:max(20, tau // 3)]) * 1.0)
        print(f"\n{'='*70}\n{name}  (T={len(ys)}, tau*={tau})\n{'='*70}")
        df = run_episode(ys, tau, alpha, eta, temperature, decay, hazard=0.01,
                          bocpd_kwargs=bocpd_kwargs, K=K, pred_window=20,
                          init_radius=init_radius, alpha_min=alpha_min, alpha_max=alpha_max)
        print(df.to_string())
        df.to_csv(out_dir / f"{name}.csv")

    print(f"\nSaved all tables to {out_dir}/")


if __name__ == "__main__":
    main()
