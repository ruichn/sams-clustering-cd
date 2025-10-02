---
title: SAMS Clustering Demo
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.28.0"
app_file: app.py
pinned: false
license: mit
---

# SAMS Clustering

This repository contains a reference implementation of the Stochastic Approximation Mean-Shift (SAMS) clustering algorithm described in

> Hyrien, O. & Baran, A. (2016). *Fast Nonparametric Density-Based Clustering of Large Data Sets Using a Stochastic Approximation Mean-Shift Algorithm*. Journal of Computational and Graphical Statistics, 25(3), 899-916.

## Quick start

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

```python
import numpy as np
from sams_clustering import SAMS, SAMSConfig

X = np.vstack([
    np.random.normal(0.0, 0.2, size=(100, 2)),
    np.random.normal(3.5, 0.2, size=(100, 2)),
])

config = SAMSConfig(bandwidth=0.7, sample_fraction=0.01, random_state=0)
model = SAMS(config)
modes, labels = model.fit(X)
```

The implementation exposes most of the hyper-parameters described in the original paper, including the two stopping rules (gradient-based and sign-change based) and Kesten-style adaptive gains.

## Testing

```bash
python -m pytest
```

## Simulation study

To reproduce a simplified version of the simulation study from the paper, run:

```bash
python -m experiments.simulation
```

The script compares the deterministic mean-shift baseline (`ρ=1.0`), subsampled SAMS
runs, the scikit-learn `MeanShift`, and a handcrafted mean-shift implementation.
Numerical summaries are printed to the console and plots (including scatter charts
comparing ground-truth vs. predicted clusters) are saved under `experiments/output/`.

### Interactive demo

Launch the Streamlit demo (compatible with Hugging Face Spaces) with:

```bash
streamlit run app.py
```

The interface exposes the same simulation controls as the CLI script, renders the
summary table, and displays all generated figures inline.
