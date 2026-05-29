"""
Run online conformal baselines + CA-AOCP on M4 Weekly data.

ALL methods share the same RollingMeanPredictor for both calibration
and online phases. The only variable is the conformal threshold mechanism.

Key design:
  - Raw M4 series → train/test split → normalize with TRAIN-ONLY min/max
  - RollingMean walk-forward on calibration split → calibration scores
  - All baselines + CA-AOCP warm-started with the SAME calibration scores
  - Online phase: shared RollingMean, shared scores, different thresholds
  - H=1 (1-step-ahead), enforced explicitly

Usage:
    uv run python experiments/run_real_data.py
"""

from __future__ import annotations
import sys, time, os, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

# ── Project path ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── CA-AOCP ───────────────────────────────────────────────────────────
from ca_aocp import CAAOCP
from ca_aocp.predictors import RollingMeanPredictor

# ── online_conformal ──────────────────────────────────────────────────
from online_conformal.split_conformal import SplitConformal
from online_conformal.nex_conformal import NExConformal
from online_conformal.faci import FACI
from online_conformal.ogd import ScaleFreeOGD
from online_conformal.saocp import SAOCP

# ── Raw M4 data (NOT the wrapper that pre-normalizes with full series) ─
from ts_datasets.forecast import M4 as RawM4


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

COVERAGE = 0.9                     # target 90% prediction intervals
MAX_SCALE = 3.0                    # max expected |y-yhat| in normalised space
N_SERIES = 10                      # number of M4 Weekly series
N_TEST = 120                       # test steps per series
CALIB_FRAC = 0.2                   # fraction of train used for calibration
PRED_WINDOW = 20                   # RollingMean window
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "output_real"

# CA-AOCP settings
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
    init_radius=1.0,    # will be recomputed from calib if available
)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def load_raw_m4_series(idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Load ONE raw (non-normalized) M4 Weekly series.

    Returns (train_raw_1d, test_raw_1d) as float64 numpy arrays.
    The split follows the M4 competition convention used by
    ``online_conformal.dataset.M4`` but WITHOUT the wrapper's
    full-series min/max normalization.
    """
    ds = RawM4("Weekly")
    ts, _ = ds[idx]                               # ts is already a pd.DataFrame
    vals = ts.iloc[:, 0].values.astype(float)     # 1-d raw values

    T = len(vals)
    # same n_test logic as online_conformal.dataset.M4 for "Weekly"
    if T > 400:
        n_test = 120
    elif T > 200:
        n_test = 60
    else:
        n_test = 26

    train_raw = vals[:-n_test]
    test_raw  = vals[-n_test:]
    return train_raw, test_raw


def normalize(train_raw: np.ndarray, test_raw: np.ndarray):
    """Normalize train and test using TRAIN-ONLY min/max. No leakage."""
    vmin = train_raw.min()
    vmax = train_raw.max()
    scale = vmax - vmin
    if scale < 1e-12:
        scale = 1.0
    train_norm = (train_raw - vmin) / scale
    test_norm  = (test_raw  - vmin) / scale
    return train_norm, test_norm, vmin, scale


def walk_forward_calibrate(
    calib_y: np.ndarray,
    *methods,
    shared_pred,
    caaocp,
) -> list[dict]:
    """Run one pass over calibration data to warm-start all methods.

    ``methods`` is a list of (name, baseline_object) tuples.
    ``shared_pred`` is the RollingMeanPredictor for baselines.
    ``caaocp`` is the CA-AOCP instance (None if not used).

    Returns a list of per-step dicts for diagnostics (may be empty).
    """
    diag = []
    for y_t in calib_y:
        yhat = shared_pred.predict(None)
        s_t = abs(y_t - yhat)
        diag.append({"y": y_t, "yhat": yhat, "score": s_t})

        # Feed score to each baseline
        for _, bl in methods:
            bl.update(
                ground_truth=pd.Series([y_t]),
                forecast=pd.Series([yhat]),
                horizon=1,
            )

        shared_pred.update(None, y_t)

        # CA-AOCP warm-start
        if caaocp is not None:
            caaocp.predict(None)       # returns (lo, hi) — discard here
            caaocp.update(None, y_t)

    return diag


# ═══════════════════════════════════════════════════════════════════════
# Core evaluation: one series
# ═══════════════════════════════════════════════════════════════════════

def evaluate_one_series(series_idx: int) -> dict | None:
    """Load, normalize, calibrate, and evaluate one M4 Weekly series."""

    # ── 1. Load raw, split, normalize ───────────────────────────────
    train_raw, test_raw = load_raw_m4_series(series_idx)
    train_norm, test_norm, vmin, scale = normalize(train_raw, test_raw)

    T_test = min(len(test_norm), N_TEST)
    test_y = test_norm[:T_test].copy()

    # ── 2. Split train into fit + calibration ───────────────────────
    n_calib = max(1, int(len(train_norm) * CALIB_FRAC))
    train_fit = train_norm[:-n_calib]
    calib_y   = train_norm[-n_calib:]

    if len(calib_y) < 2 or len(test_y) < 2:
        return None

    # ── 3. Create baselines (cold-start, no model) ──────────────────
    H = 1  # always 1-step-ahead
    baselines: list[tuple[str, object]] = [
        ("SplitCP", SplitConformal(None, None, coverage=COVERAGE)),
        ("NExCP",   NExConformal(None, None, coverage=COVERAGE)),
        ("FACI",    FACI(None, None, coverage=COVERAGE)),
        ("SF-OGD",  ScaleFreeOGD(None, None, coverage=COVERAGE, max_scale=MAX_SCALE)),
        ("SAOCP",   SAOCP(None, None, coverage=COVERAGE, max_scale=MAX_SCALE)),
    ]
    names = [n for n, _ in baselines] + ["CA-AOCP"]

    # ── 4. CA-AOCP ──────────────────────────────────────────────────
    # Estimate init_radius from train_fit scale
    init_r = float(np.std(train_fit) * 0.5)
    kwargs = dict(CAAOCP_KWARGS)
    kwargs["init_radius"] = init_r
    caaocp = CAAOCP(predictor=RollingMeanPredictor(window=PRED_WINDOW), **kwargs)

    # ── 5. Shared predictor for baselines ───────────────────────────
    shared_pred = RollingMeanPredictor(window=PRED_WINDOW)
    # Warm up with train_fit tail
    for v in train_fit[-PRED_WINDOW:]:
        shared_pred.update(None, float(v))

    # ── 6. Calibration pass (warm-start everyone) ──────────────────
    # Also warm-start CA-AOCP's internal predictor on same tail
    for v in train_fit[-PRED_WINDOW:]:
        caaocp.predict(None)        # these are dummy predictions during warmup
        caaocp.update(None, float(v))

    walk_forward_calibrate(calib_y, *baselines, shared_pred=shared_pred, caaocp=caaocp)

    # ── 7. Online loop ──────────────────────────────────────────────
    buffers = {n: {"covered": [], "width": [], "lower": [], "upper": [], "radius": []}
               for n in names}

    for t in range(len(test_y)):
        y_t = float(test_y[t])
        yhat = shared_pred.predict(None)

        for name, bl in baselines:
            delta_lb, delta_ub = bl.predict(horizon=H)
            lo = yhat + delta_lb
            hi = yhat + delta_ub
            width = hi - lo
            radius = width / 2.0
            covered = int(lo <= y_t <= hi)

            bl.update(
                ground_truth=pd.Series([y_t]),
                forecast=pd.Series([yhat]),
                horizon=H,
            )

            buffers[name]["covered"].append(covered)
            buffers[name]["width"].append(width)
            buffers[name]["lower"].append(lo)
            buffers[name]["upper"].append(hi)
            buffers[name]["radius"].append(radius)

        # CA-AOCP
        lo, hi = caaocp.predict(None)
        width = hi - lo
        radius = width / 2.0
        covered = int(lo <= y_t <= hi)
        caaocp.update(None, y_t)

        buffers["CA-AOCP"]["covered"].append(covered)
        buffers["CA-AOCP"]["width"].append(width)
        buffers["CA-AOCP"]["lower"].append(lo)
        buffers["CA-AOCP"]["upper"].append(hi)
        buffers["CA-AOCP"]["radius"].append(radius)

        shared_pred.update(None, y_t)

    # ── 8. Metrics (both normalized and raw-scale) ─────────────────
    metrics = {}
    for name in names:
        cov = np.array(buffers[name]["covered"], dtype=float)
        w_norm = np.array(buffers[name]["width"])
        metrics[name] = {
            "coverage": float(cov.mean()),
            "cov_gap": float(abs(cov.mean() - COVERAGE)),
            "mean_width_norm": float(w_norm.mean()),
            "mean_width_raw": float(w_norm.mean() * scale),   # back to original units
            "median_width_norm": float(np.median(w_norm)),
            "median_width_raw": float(np.median(w_norm) * scale),
        }
    return metrics


# ═══════════════════════════════════════════════════════════════════════
# Aggregation
# ═══════════════════════════════════════════════════════════════════════

def aggregate(all_metrics: list[dict]) -> pd.DataFrame:
    names = list(all_metrics[0].keys())
    rows = []
    for name in names:
        covs = [m[name]["coverage"] for m in all_metrics]
        gaps = [m[name]["cov_gap"] for m in all_metrics]
        w_norm = [m[name]["mean_width_norm"] for m in all_metrics]
        w_raw  = [m[name]["mean_width_raw"] for m in all_metrics]
        rows.append({
            "Method": name,
            "Coverage (mean ± std)": f"{np.mean(covs):.3f} ± {np.std(covs):.3f}",
            "Cov. Gap": f"{np.mean(gaps):.4f}",
            "Width norm (mean ± std)": f"{np.mean(w_norm):.3f} ± {np.std(w_norm):.3f}",
            "Width raw  (mean ± std)": f"{np.mean(w_raw):.1f} ± {np.std(w_raw):.1f}",
            "_cov_raw": np.mean(covs),
            "_gap_raw": np.mean(gaps),
            "_w_norm": np.mean(w_norm),
            "_w_raw": np.mean(w_raw),
        })
    return pd.DataFrame(rows).set_index("Method")


# ═══════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════

COLORS = {
    "SplitCP": "#1b9e77", "NExCP": "#d95f02", "FACI": "#7570b3",
    "SF-OGD": "#e7298a", "SAOCP": "#66a61e", "CA-AOCP": "#1f77b4",
}


def plot_aggregate_results(all_metrics: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(all_metrics[0].keys())

    covs = {n: [m[n]["coverage"] for m in all_metrics] for n in names}
    widths_norm = {n: [m[n]["mean_width_norm"] for m in all_metrics] for n in names}
    gaps = {n: [m[n]["cov_gap"] for m in all_metrics] for n in names}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    x = np.arange(len(names))

    # Coverage Gap
    gap_means = [np.mean(gaps[n]) for n in names]
    gap_stds = [np.std(gaps[n]) for n in names]
    axes[0].bar(x, gap_means, yerr=gap_stds, color=[COLORS[n] for n in names],
                capsize=4, alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("Coverage Gap")
    axes[0].set_title("Coverage Gap (lower = more accurate)")

    # Mean Width (normalized)
    w_means = [np.mean(widths_norm[n]) for n in names]
    w_stds = [np.std(widths_norm[n]) for n in names]
    axes[1].bar(x, w_means, yerr=w_stds, color=[COLORS[n] for n in names],
                capsize=4, alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("Mean Width (normalized)")
    axes[1].set_title("Mean Width — normalized scale")

    # Coverage
    cov_means = [np.mean(covs[n]) for n in names]
    cov_stds = [np.std(covs[n]) for n in names]
    axes[2].bar(x, cov_means, yerr=cov_stds, color=[COLORS[n] for n in names],
                capsize=4, alpha=0.85)
    axes[2].axhline(COVERAGE, color="black", ls="--", lw=1.2, label=f"Target {COVERAGE:.0%}")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes[2].set_ylabel("Coverage")
    axes[2].set_title("Empirical Coverage")
    axes[2].legend(fontsize=8)

    fig.suptitle(f"M4 Weekly — {len(all_metrics)} series, RollingMean calibration + online",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "01_aggregate_metrics.png", dpi=150)
    plt.close(fig)

    # Scatter: Coverage vs Width
    fig, ax = plt.subplots(figsize=(9, 7))
    for name in names:
        ax.scatter(covs[name], widths_norm[name], label=name, color=COLORS[name],
                   alpha=0.7, s=40, edgecolors="white", linewidth=0.5)
    ax.axvline(COVERAGE, color="black", ls="--", lw=1.2, alpha=0.6)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Mean Interval Width (normalized)")
    ax.set_title("Coverage vs Width (each dot = one M4 Weekly series)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "02_coverage_vs_width.png", dpi=150)
    plt.close(fig)

    print(f"Plots saved to {out_dir}/")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 65)
    print("Online Conformal Prediction — M4 Weekly (clean pipeline)")
    print("=" * 65)
    print(f"  Calibration: RollingMean walk-forward ({CALIB_FRAC:.0%} of train)")
    print(f"  Online:      RollingMean (window={PRED_WINDOW})")
    print(f"  Coverage:    {COVERAGE:.0%}")
    print(f"  Horizon:     H=1 (1-step-ahead)")
    print()

    N = min(N_SERIES, 359)   # M4 Weekly has 359 series
    all_metrics: list[dict] = []
    successes = 0
    t_start = time.perf_counter()

    for i in range(N):
        print(f"[{i+1}/{N}] ...", end=" ", flush=True)
        try:
            metrics = evaluate_one_series(series_idx=i)
        except Exception as e:
            print(f"SKIP ({type(e).__name__}: {e})")
            continue
        if metrics is not None:
            all_metrics.append(metrics)
            successes += 1
            s = metrics["FACI"]
            print(f"FACI cov={s['coverage']:.3f} width_norm={s['mean_width_norm']:.3f}")
        else:
            print("SKIP (too short)")

    elapsed = time.perf_counter() - t_start
    print(f"\nCompleted: {successes}/{N} series in {elapsed:.1f}s")

    if successes == 0:
        print("No successful evaluations.")
        return

    print("\n" + "─" * 65)
    print("Aggregate Results")
    print("─" * 65)
    agg_df = aggregate(all_metrics)
    print(agg_df[["Coverage (mean ± std)", "Cov. Gap", "Width norm (mean ± std)"]].to_string())

    print("\nGenerating plots...")
    plot_aggregate_results(all_metrics, OUTPUT_DIR)
    agg_df.to_csv(OUTPUT_DIR / "metrics.csv")
    print(f"Metrics saved to {OUTPUT_DIR / 'metrics.csv'}")
    print("Done.")


if __name__ == "__main__":
    main()
