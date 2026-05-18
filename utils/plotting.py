"""
Plotting Utilities
==================
All functions return a matplotlib Figure for use inside notebooks.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from typing import Optional

# ── Style defaults ──────────────────────────────────────────────────
COLORS = {
    "CA-AOCP": "#1f77b4",
    "ACI": "#ff7f0e",
    "ACI-SW": "#2ca02c",
    "EWMA-CP": "#d62728",
    "data": "#555555",
    "cp": "#9467bd",
    "target": "#bcbd22",
}
ALPHA_FILL = 0.15


def _cp_line(ax, tau: Optional[int], label: bool = True) -> None:
    if tau is not None:
        ax.axvline(tau, color=COLORS["cp"], lw=1.5, ls="--",
                   label="True changepoint" if label else None)


# ────────────────────────────────────────────────────────────────────


def plot_data(
    ys: np.ndarray,
    tau: Optional[int] = None,
    figsize: tuple = (12, 3),
) -> plt.Figure:
    """Plot the raw time series with an optional changepoint marker."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(ys, color=COLORS["data"], lw=0.8, alpha=0.85, label="Observations")
    _cp_line(ax, tau)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")
    ax.set_title("Synthetic Time Series with Single Changepoint")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_intervals(
    ys: np.ndarray,
    lowers: np.ndarray,
    uppers: np.ndarray,
    tau: Optional[int] = None,
    method_name: str = "CA-AOCP",
    alpha: float = 0.1,
    figsize: tuple = (14, 4),
) -> plt.Figure:
    """Plot observations with prediction interval ribbon."""
    T = len(ys)
    ts = np.arange(T)
    color = COLORS.get(method_name, "#1f77b4")

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(ts, lowers, uppers, alpha=ALPHA_FILL, color=color,
                    label=f"{method_name} {int((1-alpha)*100)}% interval")
    ax.plot(ts, lowers, lw=0.6, color=color, alpha=0.6)
    ax.plot(ts, uppers, lw=0.6, color=color, alpha=0.6)
    ax.scatter(ts, ys, s=3, color=COLORS["data"], alpha=0.5, label="Observations", zorder=3)
    _cp_line(ax, tau)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value / Interval")
    ax.set_title(f"{method_name}: Prediction Intervals")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_rolling_coverage(
    results_dict: dict,
    alpha: float = 0.1,
    tau: Optional[int] = None,
    window: int = 50,
    figsize: tuple = (12, 4),
) -> plt.Figure:
    """Rolling empirical coverage for multiple methods."""
    fig, ax = plt.subplots(figsize=figsize)
    for name, res in results_dict.items():
        cov = np.array(res.coverages if hasattr(res, "coverages") else res.coverage, dtype=float)
        rolling = np.convolve(cov, np.ones(window) / window, mode="valid")
        ts = np.arange(window - 1, len(cov))
        ax.plot(ts, rolling, label=name, color=COLORS.get(name, None), lw=1.8)
    ax.axhline(1 - alpha, color=COLORS["target"], ls="--", lw=1.5,
               label=f"Target {int((1-alpha)*100)}%")
    _cp_line(ax, tau)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xlabel("Time step")
    ax.set_ylabel(f"Empirical coverage (window={window})")
    ax.set_title("Rolling Coverage Comparison")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_bocpd_posterior(
    posteriors: list[np.ndarray],
    tau: Optional[int] = None,
    max_rl: int = 100,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """Heatmap of the BOCPD run-length posterior over time."""
    T = len(posteriors)
    grid = np.zeros((max_rl, T))
    for t, post in enumerate(posteriors):
        n = min(len(post), max_rl)
        grid[:n, t] = post[:n]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        grid, aspect="auto", origin="lower",
        extent=[0, T, 0, max_rl],
        cmap="Blues", interpolation="nearest",
    )
    if tau is not None:
        ax.axvline(tau, color=COLORS["cp"], lw=1.5, ls="--", label="True changepoint")
    plt.colorbar(im, ax=ax, label="Posterior probability")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Run length k")
    ax.set_title("BOCPD Run-Length Posterior P(r_t = k | S_{1:t})")
    if tau is not None:
        ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_regime_relevance(
    pi_list: list[np.ndarray],
    tau: Optional[int] = None,
    figsize: tuple = (14, 4),
) -> plt.Figure:
    """Plot the max and mean regime relevance score π_{t,i} over time."""
    pi_max = [p.max() if len(p) > 0 else 1.0 for p in pi_list]
    pi_mean = [p.mean() if len(p) > 0 else 1.0 for p in pi_list]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(pi_max, label="max π_{t,i}", color=COLORS["CA-AOCP"], lw=1.5)
    ax.plot(pi_mean, label="mean π_{t,i}", color=COLORS["ACI"], lw=1.5, ls="--")
    _cp_line(ax, tau)
    ax.set_xlabel("Time step")
    ax.set_ylabel("π value")
    ax.set_title("Regime Relevance Scores π_{t,i}")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_neff(
    neff: np.ndarray,
    tau: Optional[int] = None,
    figsize: tuple = (12, 3),
) -> plt.Figure:
    """Effective sample size N_eff(t) over time."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(neff, color=COLORS["CA-AOCP"], lw=1.5)
    _cp_line(ax, tau)
    ax.set_xlabel("Time step")
    ax.set_ylabel("N_eff(t)")
    ax.set_title("Effective Sample Size Over Time")
    fig.tight_layout()
    return fig


def plot_alpha_trajectory(
    results_dict: dict,
    alpha: float = 0.1,
    tau: Optional[int] = None,
    figsize: tuple = (12, 3),
) -> plt.Figure:
    """α_t trajectory for multiple methods."""
    fig, ax = plt.subplots(figsize=figsize)
    for name, res in results_dict.items():
        aseq = np.array(res.alpha_sequence if hasattr(res, "alpha_sequence") else res.alpha_seq)
        ax.plot(aseq, label=name, color=COLORS.get(name, None), lw=1.5)
    ax.axhline(alpha, color=COLORS["target"], ls="--", lw=1.2, label=f"α = {alpha}")
    _cp_line(ax, tau)
    ax.set_xlabel("Time step")
    ax.set_ylabel("α_t")
    ax.set_title("Effective Miscoverage Level α_t")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_comparison(
    ys: np.ndarray,
    results_dict: dict,
    alpha: float = 0.1,
    tau: Optional[int] = None,
    window: int = 50,
    figsize: tuple = (14, 10),
) -> plt.Figure:
    """4-panel comparison: data, coverage, interval widths, α trajectory."""
    T = len(ys)
    ts = np.arange(T)

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    # Panel 1: Data
    axes[0].plot(ys, color=COLORS["data"], lw=0.8, alpha=0.8)
    _cp_line(axes[0], tau)
    axes[0].set_ylabel("Value")
    axes[0].set_title("Time Series with Changepoint")

    # Panel 2: Rolling coverage
    for name, res in results_dict.items():
        cov = np.array(res.coverages if hasattr(res, "coverages") else res.coverage, dtype=float)
        rolling = np.convolve(cov, np.ones(window) / window, mode="valid")
        axes[1].plot(np.arange(window - 1, T), rolling,
                     label=name, color=COLORS.get(name, None), lw=1.8)
    axes[1].axhline(1 - alpha, color=COLORS["target"], ls="--", lw=1.2,
                    label=f"Target {int((1-alpha)*100)}%")
    _cp_line(axes[1], tau, label=False)
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    axes[1].set_ylabel(f"Coverage (w={window})")
    axes[1].legend(fontsize=8, ncol=2)

    # Panel 3: Interval widths
    for name, res in results_dict.items():
        lows = np.array(res.lowers if hasattr(res, "lowers") else res.lower_bounds)
        highs = np.array(res.uppers if hasattr(res, "uppers") else res.upper_bounds)
        widths = highs - lows
        # Smooth for readability
        sm = np.convolve(widths, np.ones(10) / 10, mode="valid")
        axes[2].plot(np.arange(9, T), sm, label=name,
                     color=COLORS.get(name, None), lw=1.5)
    _cp_line(axes[2], tau, label=False)
    axes[2].set_ylabel("Interval width")

    # Panel 4: Alpha trajectory
    for name, res in results_dict.items():
        aseq = np.array(res.alpha_sequence if hasattr(res, "alpha_sequence") else res.alpha_seq)
        axes[3].plot(aseq, label=name, color=COLORS.get(name, None), lw=1.5)
    axes[3].axhline(alpha, color=COLORS["target"], ls="--", lw=1.2, label=f"α = {alpha}")
    _cp_line(axes[3], tau, label=False)
    axes[3].set_ylabel("α_t")
    axes[3].set_xlabel("Time step")

    fig.tight_layout()
    return fig
