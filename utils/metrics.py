"""
Evaluation Metrics
==================
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def rolling_coverage(covered: np.ndarray, window: int = 50) -> np.ndarray:
    """Rolling empirical coverage rate."""
    c = covered.astype(float)
    return np.convolve(c, np.ones(window) / window, mode="valid")


def interval_width(lowers: np.ndarray, uppers: np.ndarray) -> np.ndarray:
    """Interval widths |upper - lower|."""
    return uppers - lowers


def winkler_score(
    ys: np.ndarray,
    lowers: np.ndarray,
    uppers: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Winkler score (width + penalty for non-coverage).

    Lower is better.  Used to compare sharpness vs coverage jointly.
    """
    widths = uppers - lowers
    penalty = (2.0 / alpha) * np.where(
        ys < lowers, lowers - ys,
        np.where(ys > uppers, ys - uppers, 0.0),
    )
    return widths + penalty


def coverage_gap(covered: np.ndarray, alpha: float) -> float:
    """Mean absolute gap from the nominal level: E[|cov - (1-α)|]."""
    return float(abs(covered.mean() - (1.0 - alpha)))


def summary_table(results_dict: dict, ys: np.ndarray, alpha: float) -> pd.DataFrame:
    """Create a summary DataFrame comparing methods.

    Parameters
    ----------
    results_dict : dict[str, BaselineResults | CAOCPResults]
        Mapping from method name to result object.
    ys : np.ndarray
        True observations Y_1, …, Y_T.
    alpha : float
        Nominal miscoverage level.
    """
    rows = []
    for name, res in results_dict.items():
        cov = np.array(res.coverages if hasattr(res, "coverages") else res.coverage)
        lows = np.array(res.lowers if hasattr(res, "lowers") else res.lower_bounds)
        highs = np.array(res.uppers if hasattr(res, "uppers") else res.upper_bounds)
        winkler = winkler_score(ys, lows, highs, alpha).mean()
        rows.append({
            "Method": name,
            "Coverage": f"{cov.mean():.3f}",
            "Target": f"{1 - alpha:.3f}",
            "Cov. Gap": f"{coverage_gap(cov, alpha):.4f}",
            "Mean Width": f"{interval_width(lows, highs).mean():.3f}",
            "Winkler": f"{winkler:.3f}",
        })
    return pd.DataFrame(rows).set_index("Method")
