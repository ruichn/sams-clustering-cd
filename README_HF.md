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

# SAMS: Stochastic Approximation Mean-Shift Clustering

This interactive demo reproduces the core ideas from Hyrien & Baran (2016). Generate synthetic datasets, tune SAMS parameters, and compare outcomes with deterministic mean-shift baselines.

## Reference

> Hyrien, O. & Baran, A. (2016). *Fast Nonparametric Density-Based Clustering of Large Data Sets Using a Stochastic Approximation Mean-Shift Algorithm*. Journal of Computational and Graphical Statistics, 25(3), 899-916.

## Features

- **Dataset Generation**: Gaussian Blobs and Mixed Densities
- **Bandwidth Control**: Auto (adaptive), Auto (Silverman), or Manual
- **Sampling Modes**: Fraction-based or Fixed size sampling
- **Baseline Comparison**: Compare with sklearn and custom MeanShift implementations
- **Interactive Visualizations**: See clustering results with ground truth comparison

## Usage

1. Configure dataset parameters in the sidebar
2. Set SAMS parameters (bandwidth, sampling mode)
3. Choose baseline algorithms to compare
4. Click "Run experiment" to see results

The app displays:
- Performance summary table with metrics (ARI, NMI, Silhouette)
- Runtime comparison chart
- Clustering visualizations including ground truth labels
