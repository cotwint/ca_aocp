"""
Run all online conformal prediction baselines + CA-AOCP on synthetic data.

Baselines from `online_conformal`:
  - SplitConformal  : empirical quantile over all history
  - NExConformal    : exponentially weighted quantile
  - FACI            : multi-learning-rate adaptive
  - SF-OGD          : scale-free online gradient descent (pinball loss)
  - SAOCP           : SF-OGD experts with coin-betting weighting

Our method:
  - CA-AOCP         : BOCPD changepoint-aware weights + smooth ACI

Usage:
    uv run python experiments/run_baselines.py
"""

from __future__ import annotations
import os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend — safe for CI / no-display machines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Project imports ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ca_aocp import CAAOCP
from ca_aocp.predictors import RollingMeanPredictor

# ── online_conformal imports ────────────────────────────────────────────
from online_conformal.split_conformal import SplitConformal
from online_conformal.nex_conformal import NExConformal
from online_conformal.faci import FACI
from online_conformal.ogd import ScaleFreeOGD
from online_conformal.saocp import SAOCP


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

DATA_PATH = PROJECT_ROOT / "data" / "synthetic_single_cp.csv"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "output"
COVERAGE = 0.9  # target coverage → 90% prediction intervals
MAX_SCALE = 10.0  # expected maximum |y - yhat|
TAU = 500  # true changepoint (for diagnostic plots only)

# CA-AOCP hyperparameters (matching the notebook)
CAAOCP_KWARGS = dict(
    alpha=1 - COVERAGE,
    eta=0.02,
    decay=0.01,
    temperature=0.5,
    hazard=0.005,
    bocpd_mu0=0.0,
    bocpd_kappa0=1.0,
    bocpd_alpha0=2.0,
    bocpd_beta0=1.0,
    max_run_length=400,
    init_radius=1.0,
)


# ═══════════════════════════════════════════════════════════════════════
# Shared predictor factory (all methods use the same architecture)
# ═══════════════════════════════════════════════════════════════════════

def make_predictor() -> RollingMeanPredictor:
    """Rolling mean over last 20 observations."""
    return RollingMeanPredictor(window=20)


# ═══════════════════════════════════════════════════════════════════════
# Experiment runner
# ═══════════════════════════════════════════════════════════════════════

def run_experiment(
    ys: np.ndarray,
    coverage: float = COVERAGE,
    max_scale: float = MAX_SCALE,
) -> dict:
    """Run all methods on a data stream and return per-step metrics.

    Parameters
    ----------
    ys : np.ndarray, shape (T,)
        Observations Y_1, …, Y_T.
    coverage : float
        Target coverage rate (default 0.9 → 90% intervals).
    max_scale : float
        Maximum expected |y − yhat|, used by SF-OGD / SAOCP.

    Returns
    -------
    results : dict[str, dict]
        Keys are method names, values are dicts with:
          - "covered" : list[bool] (1 if Y_t in interval)
          - "width"   : list[float] (interval width = 2 * radius)
          - "lower"   : list[float]
          - "upper"   : list[float]
          - "radius"  : list[float]
          - "time_s"  : total wall-clock seconds
    """
    # ── Instantiate all methods ──────────────────────────────────────
    methods: dict[str, object] = {}

    # online_conformal baselines — model=None bypasses merlion's forecaster
    methods["SplitCP"] = SplitConformal(None, None, coverage=coverage)
    methods["NExCP"] = NExConformal(None, None, coverage=coverage)
    methods["FACI"] = FACI(None, None, coverage=coverage)

    # SF-OGD and SAOCP need max_scale for learning-rate initialisation
    methods["SF-OGD"] = ScaleFreeOGD(None, None, coverage=coverage, max_scale=max_scale)
    methods["SAOCP"] = SAOCP(None, None, coverage=coverage, max_scale=max_scale)

    # CA-AOCP (our method) — uses its own predictor internally
    caaocp = CAAOCP(
        predictor=make_predictor(),
        **CAAOCP_KWARGS,
    )
    methods["CA-AOCP"] = caaocp

    # ── Shared predictor for baselines ───────────────────────────────
    shared_pred = make_predictor()

    # ── Initialize result buffers ────────────────────────────────────
    names = list(methods.keys())
    buffers: dict[str, dict] = {
        name: {
            "covered": [],
            "width": [],
            "lower": [],
            "upper": [],
            "radius": [],
            "time_s": 0.0,
        }
        for name in names
    }

    # ── Online loop ──────────────────────────────────────────────────
    T = len(ys)
    for t_idx in range(T):
        y_t = float(ys[t_idx])
        x_t = None  # univariate time series

        # Shared prediction for baselines
        yhat_shared = shared_pred.predict(x_t)

        for name in names:
            method = methods[name]

            if name == "CA-AOCP":
                # CA-AOCP: predict → get [lower, upper] before observing Y_t
                t_start = time.perf_counter()
                lo, hi = method.predict(x_t)
                # update internal state (includes score computation and BOCPD)
                method.update(x_t, y_t)
                elapsed = time.perf_counter() - t_start
                radius = (hi - lo) / 2.0
                covered = int(lo <= y_t <= hi)
            else:
                # Baselines: predict → get [lower, upper] before observing Y_t
                t_start = time.perf_counter()
                delta_lb, delta_ub = method.predict(horizon=1)
                lo = yhat_shared + delta_lb
                hi = yhat_shared + delta_ub
                radius = (hi - lo) / 2.0
                covered = int(lo <= y_t <= hi)

                # Update with true residual
                method.update(
                    ground_truth=pd.Series([y_t]),
                    forecast=pd.Series([yhat_shared]),
                    horizon=1,
                )
                elapsed = time.perf_counter() - t_start

            buffers[name]["covered"].append(covered)
            buffers[name]["width"].append(2.0 * radius)
            buffers[name]["lower"].append(lo)
            buffers[name]["upper"].append(hi)
            buffers[name]["radius"].append(radius)
            buffers[name]["time_s"] += elapsed

        # Update shared predictor for baselines
        shared_pred.update(x_t, y_t)

    return buffers


# ═══════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════

def compute_metrics(buffers: dict, alpha: float = 0.1) -> pd.DataFrame:
    """Compute per-method summary metrics."""
    rows = []
    for name, b in buffers.items():
        cov_arr = np.array(b["covered"], dtype=float)
        width_arr = np.array(b["width"])
        rows.append({
            "Method": name,
            "Coverage": cov_arr.mean(),
            "Target": 1.0 - alpha,
            "Cov. Gap": abs(cov_arr.mean() - (1.0 - alpha)),
            "Mean Width": width_arr.mean(),
            "Median Width": float(np.median(width_arr)),
            "Time (s)": b["time_s"],
        })
    return pd.DataFrame(rows).set_index("Method")


def rolling_coverage(covered: np.ndarray, window: int = 50) -> np.ndarray:
    """Rolling empirical coverage."""
    c = np.array(covered, dtype=float)
    return np.convolve(c, np.ones(window) / window, mode="valid")


# ═══════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════

COLORS: dict[str, str] = {
    "SplitCP": "#1b9e77",
    "NExCP": "#d95f02",
    "FACI": "#7570b3",
    "SF-OGD": "#e7298a",
    "SAOCP": "#66a61e",
    "CA-AOCP": "#1f77b4",
}
ALPHA_FILL = 0.12
ALPHA_TARGET = 1.0 - COVERAGE


def plot_all(buffers: dict, ys: np.ndarray, out_dir: Path) -> None:
    """Generate and save all comparison plots."""
    out_dir.mkdir(parents=True, exist_ok=True)
    T = len(ys)
    names = list(buffers.keys())

    # ── 1. Rolling coverage comparison ───────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 4.5))
    window = 50
    for name in names:
        cov = np.array(buffers[name]["covered"], dtype=float)
        roll = rolling_coverage(cov, window)
        ts = np.arange(window - 1, T)
        ax.plot(ts, roll, label=name, color=COLORS.get(name), lw=1.8)
    ax.axhline(ALPHA_TARGET, color="black", ls="--", lw=1.2,
               label=f"Target {ALPHA_TARGET:.0%}")
    if TAU is not None:
        ax.axvline(TAU, color="#9467bd", lw=1.5, ls="--", label="Changepoint τ=500")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(0.5, 1.02)
    ax.set_xlabel("Time step")
    ax.set_ylabel(f"Empirical coverage (window={window})")
    ax.set_title("Rolling Coverage: All Methods")
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "01_rolling_coverage.png", dpi=150)
    plt.close(fig)

    # ── 2. Post-changepoint recovery zoom ────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 4))
    zoom_window = 30
    for name in names:
        cov = np.array(buffers[name]["covered"], dtype=float)
        roll = rolling_coverage(cov, zoom_window)
        ts = np.arange(zoom_window - 1, T)
        ax.plot(ts, roll, label=name, color=COLORS.get(name), lw=2.0)
    ax.axhline(ALPHA_TARGET, color="black", ls="--", lw=1.2,
               label=f"Target {ALPHA_TARGET:.0%}")
    if TAU is not None:
        ax.axvline(TAU, color="#9467bd", lw=2, ls="--", label="Changepoint τ=500")
    ax.set_xlim(TAU - 20, TAU + 200)
    ax.set_ylim(0.4, 1.02)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xlabel("Time step")
    ax.set_ylabel(f"Empirical coverage (window={zoom_window})")
    ax.set_title("Post-Changepoint Coverage Recovery (zoomed)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "02_post_cp_recovery.png", dpi=150)
    plt.close(fig)

    # ── 3. Interval widths over time (smoothed) ──────────────────────
    fig, ax = plt.subplots(figsize=(13, 4))
    smooth = 20
    for name in names:
        widths = np.array(buffers[name]["width"])
        sm = np.convolve(widths, np.ones(smooth) / smooth, mode="valid")
        ts = np.arange(smooth - 1, T)
        ax.plot(ts, sm, label=name, color=COLORS.get(name), lw=1.5)
    if TAU is not None:
        ax.axvline(TAU, color="#9467bd", lw=1.5, ls="--", label="Changepoint")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Interval width (smoothed)")
    ax.set_title("Prediction Interval Width Over Time")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "03_width_over_time.png", dpi=150)
    plt.close(fig)

    # ── 4. Prediction intervals for CA-AOCP ──────────────────────────
    ca_data = buffers.get("CA-AOCP")
    if ca_data is not None:
        fig, ax = plt.subplots(figsize=(14, 4))
        ts = np.arange(T)
        lowers = np.array(ca_data["lower"])
        uppers = np.array(ca_data["upper"])
        ax.fill_between(ts, lowers, uppers, alpha=ALPHA_FILL, color=COLORS["CA-AOCP"],
                        label="CA-AOCP 90% interval")
        ax.plot(ts, lowers, lw=0.5, color=COLORS["CA-AOCP"], alpha=0.5)
        ax.plot(ts, uppers, lw=0.5, color=COLORS["CA-AOCP"], alpha=0.5)
        ax.scatter(ts, ys, s=2, color="#555555", alpha=0.4, label="Observations", zorder=3)
        if TAU is not None:
            ax.axvline(TAU, color="#9467bd", lw=1.5, ls="--", label="Changepoint τ=500")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Value / Interval")
        ax.set_title("CA-AOCP: Prediction Intervals")
        ax.legend(loc="upper left", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / "04_ca_aocp_intervals.png", dpi=150)
        plt.close(fig)

    # ── 5. Conformal radius trajectories ─────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 4))
    for name in names:
        radii = np.array(buffers[name]["radius"])
        sm = np.convolve(radii, np.ones(smooth) / smooth, mode="valid")
        ts = np.arange(smooth - 1, T)
        ax.plot(ts, sm, label=name, color=COLORS.get(name), lw=1.5)
    if TAU is not None:
        ax.axvline(TAU, color="#9467bd", lw=1.5, ls="--", label="Changepoint")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Conformal radius s_hat (smoothed)")
    ax.set_title("Conformal Radius s_hat Over Time (smoothed w=20)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "05_radius_trajectory.png", dpi=150)
    plt.close(fig)

    # ── 6. CA-AOCP pre/post score distribution ───────────────────────
    caaocp_radii = np.array(ca_data["radius"]) if ca_data else None
    if caaocp_radii is not None:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
        pre_scores = caaocp_radii[:TAU]
        post_scores = caaocp_radii[TAU:]
        axes[0].hist(pre_scores, bins=40, color=COLORS["CA-AOCP"], alpha=0.7,
                     label=f"Pre-change (t<{TAU})", density=True)
        axes[0].set_xlabel("Conformal radius q_t")
        axes[0].set_ylabel("Density")
        axes[0].set_title("Radius Distribution: Pre-Change")
        axes[0].legend()
        axes[1].hist(post_scores, bins=40, color="#d62728", alpha=0.7,
                     label=f"Post-change (t≥{TAU})", density=True)
        axes[1].set_xlabel("Conformal radius q_t")
        axes[1].set_title("Radius Distribution: Post-Change")
        axes[1].legend()
        fig.suptitle("CA-AOCP: Conformal Radius Before vs After Changepoint")
        fig.tight_layout()
        fig.savefig(out_dir / "06_radius_distribution.png", dpi=150)
        plt.close(fig)

    print(f"Plots saved to {out_dir}/")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 65)
    print("Online Conformal Prediction — Baseline Comparison")
    print("=" * 65)

    # ── Load data ────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    ys = df["value"].values
    T = len(ys)
    alpha = 1.0 - COVERAGE

    print(f"\nData: {DATA_PATH}")
    print(f"T = {T}  |  Changepoint at τ = {TAU} (for reference)")
    print(f"Pre-change  μ ≈ {ys[:TAU].mean():.3f}  σ ≈ {ys[:TAU].std():.3f}")
    print(f"Post-change μ ≈ {ys[TAU:].mean():.3f}  σ ≈ {ys[TAU:].std():.3f}")
    print(f"\nTarget coverage: {COVERAGE:.0%}  (α = {alpha})")
    print(f"Max scale (SF-OGD / SAOCP): {MAX_SCALE}")
    print(f"CA-AOCP kwargs: {CAAOCP_KWARGS}")

    # ── Run experiment ───────────────────────────────────────────────
    print("\nRunning experiment...")
    t0 = time.perf_counter()
    buffers = run_experiment(ys, coverage=COVERAGE, max_scale=MAX_SCALE)
    total_elapsed = time.perf_counter() - t0
    print(f"Done in {total_elapsed:.1f}s.\n")

    # ── Metrics ──────────────────────────────────────────────────────
    metrics_df = compute_metrics(buffers, alpha=alpha)
    # Reorder columns for readability
    metrics_df = metrics_df[["Coverage", "Target", "Cov. Gap", "Mean Width", "Median Width", "Time (s)"]]

    # Also compute pre- and post-changepoint coverage
    pre_cov = {}
    post_cov = {}
    for name in buffers:
        covered = np.array(buffers[name]["covered"], dtype=float)
        pre_cov[name] = covered[:TAU].mean()
        post_cov[name] = covered[TAU:].mean()

    print("─" * 65)
    print("Summary Metrics")
    print("─" * 65)
    for name in buffers:
        row = metrics_df.loc[name]
        print(
            f"  {name:<12} "
            f"Coverage={row['Coverage']:.4f}  "
            f"Gap={row['Cov. Gap']:.4f}  "
            f"Width={row['Mean Width']:.3f}  "
            f"Time={row['Time (s)']:.2f}s"
        )
    print()

    print("─" * 65)
    print("Pre-Change vs Post-Change Coverage")
    print("─" * 65)
    for name in buffers:
        print(f"  {name:<12} pre-τ: {pre_cov[name]:.4f}   post-τ: {post_cov[name]:.4f}")

    # ── Plot ─────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_all(buffers, ys, OUTPUT_DIR)

    # ── Save metrics CSV ─────────────────────────────────────────────
    metrics_df.to_csv(OUTPUT_DIR / "metrics.csv")
    print(f"Metrics saved to {OUTPUT_DIR / 'metrics.csv'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
