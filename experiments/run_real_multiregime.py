"""
Real Multi-Regime Data Validation
=====================================
All prior experiments used synthetic (or paper-moment-matched-reconstructed)
data with an engineered, exactly-known mean-shift changepoint. This script
adds genuine real-world series with natural regime changes, of two
qualitatively different kinds:

  1. Financial volatility regimes (MSFT daily log returns, via pmdarima's
     bundled dataset -- no network fetch needed). Two well-documented
     historical episodes are extracted:
       - "dotcom":  2000 dot-com crash (event ~2000-03-10, NASDAQ peak)
       - "gfc":     2008 financial crisis (event ~2008-09-15, Lehman collapse)
     This is a DIFFERENT kind of distribution shift than the paper's
     benchmarks: returns are approximately mean-zero throughout, and what
     shifts is the *variance* (volatility clustering), not the mean. Since
     BOCPD's NIG model tracks both mean and variance jointly, this tests
     whether the machinery generalizes beyond mean-shift changepoints.

  2. Electricity demand (Taylor's half-hourly England/Wales demand series,
     4032 points ~ 12 weeks, via pmdarima). No single engineered changepoint;
     used as a general non-engineered real-world stress test. Caveat: strong
     daily/weekly seasonality is NOT modeled by the simple rolling-mean
     predictor (matching the paper's own predictor choice), so all methods'
     absolute numbers will look worse than on the synthetic benchmarks --
     the comparison across methods is still informative, the absolute
     coverage numbers are not directly comparable to Table 1/2 in the paper.

Methods: reuses the same 8-method ablation family as run_ablation.py.

Usage:
    python3 experiments/run_real_multiregime.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_ablation import make_methods, winkler_score


# ═══════════════════════════════════════════════════════════════════════
# Real data loaders (all offline / bundled with pmdarima -- no network)
# ═══════════════════════════════════════════════════════════════════════

def load_msft_returns(start: str, end: str) -> tuple[np.ndarray, pd.Series]:
    import pmdarima as pm
    msft = pm.datasets.load_msft()
    msft["Date"] = pd.to_datetime(msft["Date"])
    sub = msft[(msft.Date >= start) & (msft.Date <= end)].reset_index(drop=True)
    price = sub["Close"].values.astype(float)
    logret = np.diff(np.log(price)) * 100.0  # percent log returns
    dates = sub["Date"].values[1:]
    return logret, pd.Series(dates)


def load_taylor_electricity() -> np.ndarray:
    import pmdarima as pm
    y = np.asarray(pm.datasets.load_taylor(), dtype=float)
    return y


EPISODES = {
    "MSFT_dotcom_2000": dict(start="1999-06-01", end="2003-06-01", event="2000-03-10"),
    "MSFT_gfc_2008":    dict(start="2006-06-01", end="2010-06-01", event="2008-09-15"),
}


# ═══════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════

def summarize(name, res, ys, alpha, event_idx=None):
    cov = res.covered if hasattr(res, "covered") else res["covered"]
    lo = res.lowers if hasattr(res, "lowers") else res["lower"]
    hi = res.uppers if hasattr(res, "uppers") else res["upper"]
    width = res.radii * 2 if hasattr(res, "radii") else res["width"]
    cov = np.asarray(cov, dtype=float)
    lo = np.asarray(lo); hi = np.asarray(hi); width = np.asarray(width)
    w = winkler_score(np.asarray(ys), lo, hi, alpha)

    row = {
        "Method": name,
        "coverage": cov.mean(),
        "cov_gap": abs(cov.mean() - (1 - alpha)),
        "mean_width": width.mean(),
        "winkler": w.mean(),
    }
    if event_idx is not None:
        T = len(ys)
        for h in (10, 25, 50, 100):
            end = min(event_idx + h, T)
            seg = cov[event_idx:end]
            row[f"postcov{h}"] = seg.mean() if len(seg) else np.nan
    return row


def run_on_series(ys, alpha, eta, temperature, decay, hazard, bocpd_kwargs, K,
                   pred_window, init_radius, alpha_min, alpha_max, event_idx=None):
    methods = make_methods(alpha, eta, temperature, decay, hazard, bocpd_kwargs, K,
                            pred_window, init_radius, alpha_min, alpha_max)
    rows = []
    for mname, m in methods.items():
        res = m.run(ys)
        rows.append(summarize(mname, res, ys, alpha, event_idx))
    return pd.DataFrame(rows).set_index("Method")


def main():
    alpha = 0.1
    eta = 0.02
    temperature = 0.5
    decay = 0.01
    bocpd_kwargs = dict(mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=1.0)
    K = 50
    alpha_min, alpha_max = alpha / 2, (1 + alpha) / 2

    out_dir = PROJECT_ROOT / "experiments" / "output_real_multiregime"
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.set_option("display.width", 180)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    # -- Financial volatility episodes --
    for name, cfg in EPISODES.items():
        ys, dates = load_msft_returns(cfg["start"], cfg["end"])
        event_ts = pd.Timestamp(cfg["event"])
        event_idx = int(np.searchsorted(dates.values, np.datetime64(event_ts)))
        init_radius = float(np.std(ys[:100]) * 1.5)

        print(f"\n{'='*70}\n{name}  (T={len(ys)}, event_idx={event_idx}, event_date={cfg['event']})\n{'='*70}")
        df = run_on_series(ys, alpha, eta, temperature, decay, hazard=0.01,
                            bocpd_kwargs=bocpd_kwargs, K=K, pred_window=20,
                            init_radius=init_radius, alpha_min=alpha_min, alpha_max=alpha_max,
                            event_idx=event_idx)
        print(df.to_string())
        df.to_csv(out_dir / f"{name}.csv")

    # -- Electricity demand (no engineered event date) --
    ys_elec = load_taylor_electricity()
    init_radius = float(np.std(ys_elec[:200]) * 0.5)
    print(f"\n{'='*70}\nTaylor_electricity_demand  (T={len(ys_elec)})\n{'='*70}")
    df_elec = run_on_series(ys_elec, alpha, eta, temperature, decay, hazard=0.005,
                             bocpd_kwargs=bocpd_kwargs, K=K, pred_window=48,
                             init_radius=init_radius, alpha_min=alpha_min, alpha_max=alpha_max,
                             event_idx=None)
    print(df_elec.to_string())
    df_elec.to_csv(out_dir / "Taylor_electricity_demand.csv")

    print(f"\nSaved all tables to {out_dir}/")


if __name__ == "__main__":
    main()
