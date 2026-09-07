"""
Component Ablation Study for CA-AOCP
=====================================
Isolates the contribution of each of CA-AOCP's three components:

    w_{t,i} = pi_{t,i} (BOCPD regime relevance)  *  exp(-lambda(t-i)) (recency decay)
    + ACI adaptive-alpha update (residual calibration correction)

Methods compared (8 total):
    ACI                  : no BOCPD, no decay,        ACI adaptive
    EWMA-ACI              : no BOCPD, decay,           ACI adaptive       [= EWMA-CP baseline]
    BOCPD-weighted CP     : BOCPD,    no decay,        alpha fixed (no ACI)
    BOCPD-weighted ACI    : BOCPD,    no decay,        ACI adaptive
    CA-AOCP w/o ACI       : BOCPD,    decay,           alpha fixed (no ACI)
    Full CA-AOCP          : BOCPD,    decay,           ACI adaptive       [= paper's method]
    Hard-reset CP         : BOCPD MAP changepoint reset, uniform window,  alpha fixed
    Hard-reset ACI        : BOCPD MAP changepoint reset, uniform window,  ACI adaptive

Run on both Synthetic-CP (existing data/synthetic_single_cp.csv, T=1000, tau*=500)
and a QC-matched synthetic series rebuilt from the paper's Table 8 statistics
(T=313, tau*=148, N(0.67,1.16^2) -> N(4.33,1.01^2) + drift 0.009/step).

Usage:
    python3 experiments/run_ablation.py
"""

from __future__ import annotations
import sys
from collections import deque
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ca_aocp.bocpd import GaussianBOCPD
from ca_aocp.conformal import weighted_quantile, prediction_interval, coverage_indicator
from ca_aocp.adaptive_update import smooth_surrogate, adaptive_update
from ca_aocp.predictors import RollingMeanPredictor


# ═══════════════════════════════════════════════════════════════════════
# Generic method covering all 8 ablation cells
# ═══════════════════════════════════════════════════════════════════════

class GenericMethod:
    """Unifies ACI / EWMA-ACI / BOCPD-weighted / CA-AOCP / hard-reset variants.

    Parameters
    ----------
    use_bocpd : bool
        If True, weight includes the BOCPD regime-relevance posterior pi_{t,i}.
        If False, pi_{t,i} == 1 for all i (uniform regime relevance).
    decay : float
        Recency decay lambda >= 0.  0 disables the exponential forgetting factor.
    adapt_alpha : bool
        If True, run the ACI projected-gradient update on alpha_t.
        If False, alpha_t is held fixed at the nominal level (no ACI correction).
    hard_reset : bool
        If True, ignore soft BOCPD weights entirely; instead use the BOCPD
        posterior only to obtain a MAP run-length estimate r_hat, and take a
        UNIFORM weight over the most recent (r_hat + 1) scores (Appendix B.2(iv)
        degenerate case: BOCPD posterior collapsed to a point estimate).
    """

    def __init__(
        self,
        alpha: float = 0.1,
        eta: float = 0.02,
        temperature: float = 0.5,
        decay: float = 0.0,
        hazard: float = 0.01,
        bocpd_kwargs: dict | None = None,
        max_run_length: int = 50,
        use_bocpd: bool = False,
        adapt_alpha: bool = True,
        hard_reset: bool = False,
        predictor=None,
        init_radius: float = 1.0,
        alpha_min: float | None = None,
        alpha_max: float | None = None,
    ) -> None:
        self.alpha = alpha
        self.eta = eta
        self.temperature = temperature
        self.decay = decay
        self.use_bocpd = use_bocpd
        self.adapt_alpha = adapt_alpha
        self.hard_reset = hard_reset
        self.alpha_min = alpha_min if alpha_min is not None else alpha / 2.0
        self.alpha_max = alpha_max if alpha_max is not None else (1.0 + alpha) / 2.0

        self._alpha_t = alpha
        self._q_t = init_radius
        # (2026-08 follow-up) Only BOCPD-driven variants (use_bocpd or
        # hard_reset) get a K-bounded score buffer here. That matches the
        # paper's own hyperparameter table (Table 9): truncation depth K is
        # listed under "CA-AOCP only" -- ACI and EWMA-ACI (the plain-ACI and
        # EWMA-CP baselines) are NOT given a K in the paper and are meant to
        # stay full-history (ACI, Table 6: O(t)) or decay-only (EWMA-CP,
        # Table 6: O(1) amortised), so their buffers are left unbounded,
        # unchanged from before. Before this fix, ALL eight ablation cells
        # used an unbounded list here regardless of use_bocpd/hard_reset, so
        # every BOCPD-driven variant's calibration set silently included the
        # entire history rather than the K most recent scores -- meaning the
        # cov_gap/width/Winkler numbers reported for those six methods in
        # the Task #1-#6 experiment summary were never actually measuring
        # K-truncated CA-AOCP-style calibration.
        self._max_run_length = max_run_length
        self._truncated = use_bocpd or hard_reset
        self._scores: deque[float] | list[float] = (
            deque(maxlen=max_run_length) if self._truncated else []
        )
        self._pi: np.ndarray | None = None  # pi_{t,i} snapshot to use for THIS step's q_t (already baked into _q_t)
        self._predictor = predictor or RollingMeanPredictor(window=20)

        needs_bocpd = use_bocpd or hard_reset
        bk = bocpd_kwargs or {}
        self._bocpd = GaussianBOCPD(hazard=hazard, max_run_length=max_run_length, **bk) if needs_bocpd else None

    # -- weight construction for the CURRENT calibration set --------------
    def _make_weights(self, pi: np.ndarray | None, n: int) -> np.ndarray:
        """n = number of historical scores available (S_1..S_n)."""
        if self.hard_reset:
            # pi here is actually the run-length posterior array (prob per k)
            run_post = pi
            r_hat = int(np.argmax(run_post)) if run_post is not None else n - 1
            window = min(r_hat + 1, n)
            w = np.zeros(n)
            w[-window:] = 1.0 / window
            return w

        lags = np.arange(n, 0, -1, dtype=float)  # lags[i-1] = n+1-i  (age of obs i, most recent=1)
        log_decay = -self.decay * lags if self.decay > 0 else np.zeros(n)

        if self.use_bocpd and pi is not None:
            log_pi = np.log(np.clip(pi, 1e-300, None))
        else:
            log_pi = np.zeros(n)

        log_w = log_pi + log_decay
        log_w -= log_w.max()
        w = np.exp(log_w)
        total = w.sum()
        if total < 1e-300:
            return np.ones(n) / n
        return w / total

    def run(self, ys: np.ndarray) -> dict:
        T = len(ys)
        covered = np.zeros(T, dtype=int)
        widths = np.zeros(T)
        lowers = np.zeros(T)
        uppers = np.zeros(T)
        radii = np.zeros(T)
        alpha_seq = np.zeros(T)
        r_hat_seq = np.full(T, np.nan)  # MAP run-length estimate at each step (BOCPD-based methods only)

        for t_idx in range(T):
            y_t = float(ys[t_idx])
            yhat = self._predictor.predict(None)
            lo, hi = prediction_interval(yhat, self._q_t)
            covered[t_idx] = coverage_indicator(y_t, lo, hi)
            widths[t_idx] = hi - lo
            lowers[t_idx] = lo
            uppers[t_idx] = hi
            radii[t_idx] = self._q_t
            alpha_seq[t_idx] = self._alpha_t

            s_t = abs(y_t - yhat)
            self._scores.append(s_t)

            ell_t = smooth_surrogate(s_t, self._q_t, self.temperature)
            if self.adapt_alpha:
                alpha_next = adaptive_update(
                    self._alpha_t, ell_t, self.alpha, self.eta,
                    self.alpha_min, self.alpha_max,
                )
            else:
                alpha_next = self.alpha  # fixed nominal level, no ACI correction

            # -- update BOCPD (if needed) to get info for q_{t+1} --
            if self._bocpd is not None:
                # compute_legacy_pi=False: we only ever need the bounded
                # pi_recent()/get_run_length_posterior() views below, both
                # already O(K); building the legacy O(t) pi vector here (as
                # this used to do unconditionally) would silently reintroduce
                # a per-step cost that grows with t, same leak as fixed in
                # ca_aocp.algorithm.CAAOCP (see bocpd.py's module docstring).
                self._bocpd.update(s_t, compute_legacy_pi=False)
                run_post = self._bocpd.get_run_length_posterior()  # O(K), dense but bounded
                r_hat_seq[t_idx] = int(np.argmax(run_post))
                if self.hard_reset:
                    weight_info = run_post
                else:
                    # Bounded O(K) pi window, matching the K-sized score
                    # buffer above -- was the legacy full-length pi_new from
                    # update()'s old return value, matched against an
                    # unbounded score list.
                    weight_info = self._bocpd.pi_recent(self._max_run_length)
            else:
                weight_info = None

            n = len(self._scores)
            if n >= 2:
                w = self._make_weights(weight_info, n)
                scores_arr = np.array(self._scores)
                self._q_t = weighted_quantile(scores_arr, w, alpha_next)

            self._alpha_t = alpha_next
            self._predictor.update(None, y_t)

        return {
            "covered": covered, "width": widths, "lower": lowers,
            "upper": uppers, "radius": radii, "alpha_seq": alpha_seq,
            "r_hat": r_hat_seq,
        }


# ═══════════════════════════════════════════════════════════════════════
# Method factory (the 8 ablation cells)
# ═══════════════════════════════════════════════════════════════════════

def make_methods(alpha, eta, temperature, decay, hazard, bocpd_kwargs, K,
                  pred_window, init_radius, alpha_min, alpha_max):
    def pred():
        return RollingMeanPredictor(window=pred_window)

    common = dict(alpha=alpha, eta=eta, temperature=temperature, hazard=hazard,
                   bocpd_kwargs=bocpd_kwargs, max_run_length=K,
                   init_radius=init_radius, alpha_min=alpha_min, alpha_max=alpha_max)

    return {
        "ACI":                GenericMethod(decay=0.0,   use_bocpd=False, adapt_alpha=True,  hard_reset=False, predictor=pred(), **common),
        "EWMA-ACI":            GenericMethod(decay=decay, use_bocpd=False, adapt_alpha=True,  hard_reset=False, predictor=pred(), **common),
        "BOCPD-weighted CP":   GenericMethod(decay=0.0,   use_bocpd=True,  adapt_alpha=False, hard_reset=False, predictor=pred(), **common),
        "BOCPD-weighted ACI":  GenericMethod(decay=0.0,   use_bocpd=True,  adapt_alpha=True,  hard_reset=False, predictor=pred(), **common),
        "CA-AOCP w/o ACI":     GenericMethod(decay=decay, use_bocpd=True,  adapt_alpha=False, hard_reset=False, predictor=pred(), **common),
        "Full CA-AOCP":        GenericMethod(decay=decay, use_bocpd=True,  adapt_alpha=True,  hard_reset=False, predictor=pred(), **common),
        "Hard-reset CP":       GenericMethod(decay=0.0,   use_bocpd=False, adapt_alpha=False, hard_reset=True,  predictor=pred(), **common),
        "Hard-reset ACI":      GenericMethod(decay=0.0,   use_bocpd=False, adapt_alpha=True,  hard_reset=True,  predictor=pred(), **common),
    }


# ═══════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════

def load_synthetic_cp(project_root: Path) -> tuple[np.ndarray, int]:
    df = pd.read_csv(project_root / "data" / "synthetic_single_cp.csv")
    return df["value"].values.astype(float), 500


def build_qc_dataset(seed: int = 42) -> tuple[np.ndarray, int]:
    """Rebuild the Quality-Control series from paper Table 8 statistics.

    T=313, tau*=148, N(0.67,1.16^2) -> N(4.33,1.01^2), post-change drift
    slope ~= 0.009/step.  Not the original data (not released) -- a
    seed-matched reconstruction from the reported moments, for ablation
    purposes only.
    """
    rng = np.random.default_rng(seed)
    T, tau = 313, 148
    pre = rng.normal(0.67, 1.16, tau)
    n_post = T - tau
    drift = 0.009 * np.arange(1, n_post + 1)
    post = rng.normal(4.33, 1.01, n_post) + drift
    return np.concatenate([pre, post]), tau


# ═══════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════

def winkler_score(ys, lowers, uppers, alpha):
    widths = uppers - lowers
    penalty = (2.0 / alpha) * np.where(
        ys < lowers, lowers - ys, np.where(ys > uppers, ys - uppers, 0.0),
    )
    return widths + penalty


def summarize(name, res, ys, tau, alpha):
    cov = res["covered"].astype(float)
    lo, hi = res["lower"], res["upper"]
    w = winkler_score(ys, lo, hi, alpha)

    row = {
        "Method": name,
        "Coverage": cov.mean(),
        "Cov.Gap": abs(cov.mean() - (1 - alpha)),
        "MeanWidth": res["width"].mean(),
        "Winkler": w.mean(),
    }
    # post-change local recovery windows
    for h in (10, 25, 50):
        end = min(tau + h, len(ys))
        seg = cov[tau:end]
        row[f"PostCov@{h}"] = seg.mean() if len(seg) > 0 else np.nan
        row[f"PostGap@{h}"] = abs(seg.mean() - (1 - alpha)) if len(seg) > 0 else np.nan
    return row


def run_dataset(name, ys, tau, alpha, eta, temperature, decay, hazard,
                 bocpd_kwargs, K, pred_window, init_radius, alpha_min, alpha_max):
    print(f"\n{'='*70}\n{name}  (T={len(ys)}, tau*={tau})\n{'='*70}")
    methods = make_methods(alpha, eta, temperature, decay, hazard, bocpd_kwargs,
                            K, pred_window, init_radius, alpha_min, alpha_max)
    rows = []
    for mname, m in methods.items():
        res = m.run(ys)
        rows.append(summarize(mname, res, ys, tau, alpha))
    df = pd.DataFrame(rows).set_index("Method")
    return df


def main():
    alpha = 0.1
    eta = 0.02
    temperature = 0.5
    decay = 0.01
    bocpd_kwargs = dict(mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=1.0)
    K = 50
    alpha_min, alpha_max = alpha / 2, (1 + alpha) / 2
    init_radius = 1.0

    ys_syn, tau_syn = load_synthetic_cp(PROJECT_ROOT)
    df_syn = run_dataset("Synthetic-CP", ys_syn, tau_syn, alpha, eta, temperature,
                          decay, hazard=0.005, bocpd_kwargs=bocpd_kwargs, K=K,
                          pred_window=20, init_radius=init_radius,
                          alpha_min=alpha_min, alpha_max=alpha_max)

    ys_qc, tau_qc = build_qc_dataset(seed=42)
    df_qc = run_dataset("Quality-Control (reconstructed)", ys_qc, tau_qc, alpha, eta,
                         temperature, decay, hazard=0.010, bocpd_kwargs=bocpd_kwargs, K=K,
                         pred_window=15, init_radius=init_radius,
                         alpha_min=alpha_min, alpha_max=alpha_max)

    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    print("\n\n--- Synthetic-CP ablation ---")
    print(df_syn.to_string())
    print("\n\n--- Quality-Control (reconstructed) ablation ---")
    print(df_qc.to_string())

    out_dir = PROJECT_ROOT / "experiments" / "output_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_syn.to_csv(out_dir / "ablation_synthetic_cp.csv")
    df_qc.to_csv(out_dir / "ablation_quality_control.csv")
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
