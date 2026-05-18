"""
utils/__init__.py  —  metrics and plotting utilities.
"""
from .metrics import (
    rolling_coverage,
    interval_width,
    winkler_score,
    coverage_gap,
    summary_table,
)
from .plotting import (
    plot_data,
    plot_intervals,
    plot_rolling_coverage,
    plot_bocpd_posterior,
    plot_regime_relevance,
    plot_neff,
    plot_alpha_trajectory,
    plot_comparison,
)

__all__ = [
    "rolling_coverage", "interval_width", "winkler_score",
    "coverage_gap", "summary_table",
    "plot_data", "plot_intervals", "plot_rolling_coverage",
    "plot_bocpd_posterior", "plot_regime_relevance",
    "plot_neff", "plot_alpha_trajectory", "plot_comparison",
]
