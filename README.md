# CA-AOCP: Changepoint-Aware Adaptive Online Conformal Prediction

A clean Python implementation of the CA-AOCP algorithm described in the paper
*"Changepoint-Aware Adaptive Online Conformal Prediction"*.

## Overview

CA-AOCP combines three components to guarantee valid prediction intervals
under non-exchangeable, piecewise-stationary data streams:

| Component | Role | Module |
|---|---|---|
| **BOCPD** | Computes regime relevance scores π_{t,i} | `ca_aocp/bocpd.py` |
| **Importance weights** | w_{t,i} = π_{t,i} · exp(−λ(t−i)) | `ca_aocp/weights.py` |
| **Weighted conformal calibration** | Quantile of weighted score distribution | `ca_aocp/conformal.py` |
| **Adaptive ACI update** | Tracks cumulative coverage error | `ca_aocp/adaptive_update.py` |

## Repository Structure

```
ca_aocp_repo/
├── ca_aocp/              # Core algorithm modules
│   ├── bocpd.py          # Gaussian BOCPD with NIG conjugate prior
│   ├── weights.py        # Changepoint-aware importance weights
│   ├── conformal.py      # Weighted conformal quantile
│   ├── adaptive_update.py# Smooth ACI update rule
│   ├── predictors.py     # Base predictors (rolling mean, EWMA, …)
│   └── algorithm.py      # Main CAAOCP class (Algorithm 1)
├── baselines/            # Comparison methods
│   └── __init__.py       # ACI, ACI-SW, EWMA-CP
├── utils/                # Metrics and plotting
│   ├── metrics.py        # Coverage, Winkler score, summary table
│   └── plotting.py       # All visualisation functions
├── data/
│   └── synthetic_single_cp.csv   # Toy dataset (T=1000, τ=500)
├── notebooks/
│   └── 01_demo_synthetic_cp.ipynb # Full demonstration notebook
├── requirements.txt
└── setup.py
```

## Installation

```bash
# Clone or download the repository, then:
cd ca_aocp_repo
pip install -e .

# Or directly:
pip install -r requirements.txt
```

## Quick Start

```python
import numpy as np
from ca_aocp import CAAOCP
from ca_aocp.predictors import RollingMeanPredictor

# Load data
import pandas as pd
df = pd.read_csv("data/synthetic_single_cp.csv")
ys = df["value"].values

# Configure CA-AOCP (α=0.1 → target 90% coverage)
model = CAAOCP(
    alpha=0.1,       # nominal miscoverage level
    eta=0.02,        # ACI step size
    decay=0.01,      # BOCPD weight recency decay λ
    temperature=0.5, # smooth surrogate temperature τ
    hazard=0.005,    # BOCPD changepoint prior h
    predictor=RollingMeanPredictor(window=20),
)

# Run online
results = model.run(ys)

print(f"Long-run coverage: {results.long_run_coverage():.3f}")
print(f"Mean interval width: {results.radii.mean() * 2:.3f}")
```

## Baselines

```python
from baselines import ACI, ACISlidingWindow, EWMACP

aci     = ACI(alpha=0.1, eta=0.02).run(ys)
aci_sw  = ACISlidingWindow(alpha=0.1, eta=0.02, window=100).run(ys)
ewma_cp = EWMACP(alpha=0.1, eta=0.02, decay=0.01).run(ys)
```

The checked-in multi-seed comparison against the published online-conformal
baselines is available in `experiments/run_baselines_multiseed.py`. Its raw
results and paired significance tables are kept under
`experiments/output_baselines_multiseed*/` and
`experiments/output_significance_baselines*/`.

## Formal Theoretical Result

| Result | Reference |
|---|---|---|
| Formal CA-AOCP result under the paper's stated assumptions | Proposition 1 (paper) |

## Hyperparameter Guide

| Parameter | Recommended | Role |
|---|---|---|
| `alpha` | application-specific | Nominal miscoverage α |
| `eta` | 0.01–0.05 | ACI step size |
| `decay` (λ) | 0.005–0.05 | Within-regime recency |
| `temperature` (τ) | ≍ η / M | Smooth surrogate sharpness |
| `hazard` (h) | ≍ 1/L (regime length) | BOCPD changepoint prior |
| `max_run_length` (K) | 50 (paper-reported results) | BOCPD truncation depth; larger K may be useful in specific settings. See `experiments/run_hparam_sensitivity.py` and `experiments/output_hparam/sensitivity_K.csv`. |
