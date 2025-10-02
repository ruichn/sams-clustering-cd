"""Streamlit demo for SAMS clustering inspired by the Hugging Face Space."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from experiments.simulation import map_labels_to_true, run_meanshift_custom, run_meanshift_sklearn
from sams_clustering.sams import SAMS, SAMSConfig


@dataclass
class DatasetConfig:
    name: str
    n_samples: int
    n_features: int
    random_state: int
    noise: float
    cluster_std: float
    n_clusters: int | None


def generate_dataset(cfg: DatasetConfig) -> Tuple[np.ndarray, np.ndarray]:
    if cfg.name == "Gaussian Blobs":
        centers = cfg.n_clusters or max(3, cfg.n_features + 1)
        X, y = make_blobs(
            n_samples=cfg.n_samples,
            centers=centers,
            n_features=cfg.n_features,
            cluster_std=cfg.cluster_std,
            random_state=cfg.random_state,
        )
    elif cfg.name == "Mixed Densities":
        centers = cfg.n_clusters or max(3, cfg.n_features + 1)
        n1 = cfg.n_samples // 2
        n2 = cfg.n_samples - n1
        X1, y1 = make_blobs(
            n_samples=n1,
            centers=centers,
            n_features=cfg.n_features,
            cluster_std=cfg.cluster_std,
            random_state=cfg.random_state,
        )
        X2, y2 = make_blobs(
            n_samples=n2,
            centers=max(2, centers - 1),
            n_features=cfg.n_features,
            cluster_std=float(cfg.cluster_std) * 1.8,
            random_state=cfg.random_state + 1,
        )
        y2 += y1.max() + 1
        X = np.vstack([X1, X2])
        y = np.concatenate([y1, y2])
    elif cfg.name == "Two Moons":
        base_dim = max(2, cfg.n_features)
        X, y = make_moons(n_samples=cfg.n_samples, noise=cfg.noise or 0.1, random_state=cfg.random_state)
        if base_dim > 2:
            extra = np.random.default_rng(cfg.random_state).normal(size=(cfg.n_samples, base_dim - 2))
            X = np.hstack([X, extra])
        elif base_dim == 1:
            X = X[:, :1]
        else:
            X = X[:, :2]
    elif cfg.name == "Concentric Circles":
        base_dim = max(2, cfg.n_features)
        X, y = make_circles(n_samples=cfg.n_samples, factor=0.5, noise=cfg.noise or 0.05, random_state=cfg.random_state)
        if base_dim > 2:
            extra = np.random.default_rng(cfg.random_state).normal(size=(cfg.n_samples, base_dim - 2))
            X = np.hstack([X, extra])
        elif base_dim == 1:
            X = X[:, :1]
        else:
            X = X[:, :2]
    else:
        raise ValueError(f"Unsupported dataset '{cfg.name}'")

    X = StandardScaler().fit_transform(X)
    return X.astype(float), y.astype(int)


def run_sams(
    X: np.ndarray,
    y_true: np.ndarray,
    bandwidth: float | None,
    sample_fraction: float,
    seed: int,
    bandwidth_method: str | None,
    fixed_sample_size: int | None = None,
) -> Dict[str, object]:
    cfg = SAMSConfig(
        bandwidth=bandwidth,
        bandwidth_method=bandwidth_method or "manual",
        sample_fraction=sample_fraction,
        fixed_sample_size=fixed_sample_size,
        max_iter=200,
        random_state=seed,
        merge_radius=bandwidth * 0.75 if bandwidth is not None else None,
    )
    model = SAMS(cfg)
    start = time.perf_counter()
    modes, labels = model.fit(X)
    runtime = time.perf_counter() - start
    mapped, err = map_labels_to_true(y_true, labels)
    counts = format_cluster_counts(labels)
    if fixed_sample_size is not None:
        label = f"SAMS (n={fixed_sample_size})"
    else:
        label = f"SAMS (ρ={sample_fraction:.3f})"
    if bandwidth is None:
        label += f" [{cfg.bandwidth_method}]"
    if bandwidth is None:
        bw_array = model._last_bandwidths if model._last_bandwidths is not None else model._prepare_bandwidths(X)
        bw_value = float(np.mean(bw_array))
    else:
        bw_value = float(bandwidth)
    return {
        "label": label,
        "labels": labels,
        "mapped": mapped,
        "time": runtime,
        "n_clusters": len(np.unique(labels)),
        "bandwidth": bw_value,
        "error": err,
        "modes": modes,
        "counts": counts,
    }


def compute_metrics(X: np.ndarray, y_true: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics["ARI"] = adjusted_rand_score(y_true, labels)
    metrics["NMI"] = normalized_mutual_info_score(y_true, labels)
    if len(np.unique(labels)) > 1 and X.shape[0] > len(np.unique(labels)):
        metrics["Silhouette"] = silhouette_score(X, labels)
    else:
        metrics["Silhouette"] = float("nan")
    return metrics


def ensure_2d_projection(X: np.ndarray) -> Tuple[np.ndarray, bool]:
    if X.shape[1] <= 2:
        return X[:, :2], False
    pca = PCA(n_components=2, random_state=0)
    return pca.fit_transform(X), True


def format_cluster_counts(labels: np.ndarray) -> str:
    unique, counts = np.unique(labels, return_counts=True)
    # Sort by counts in descending order
    sorted_indices = np.argsort(counts)[::-1]
    pairs = [f"{int(unique[i])}:{int(counts[i])}" for i in sorted_indices]
    return ", ".join(pairs)


def plot_clusters(X: np.ndarray, labels: np.ndarray, title: str) -> plt.Figure:
    X_proj, is_projected = ensure_2d_projection(X)
    fig, ax = plt.subplots(figsize=(5, 5))
    scatter = ax.scatter(X_proj[:, 0], X_proj[:, 1], c=labels, cmap="tab20", s=12, alpha=0.8)
    ax.set_title(title + (" (PCA)" if is_projected else ""))
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label="Cluster ID")
    fig.tight_layout()
    return fig


def plot_runtime(summary: Dict[str, Dict[str, object]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    methods = list(summary.keys())
    times = [summary[m]["time"] for m in methods]
    colors = ["#ff6b6b" if "SAMS" in m else "#4ecdc4" for m in methods]
    bars = ax.bar(methods, times, color=colors, alpha=0.85)
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.3f}s", ha="center", va="bottom")
    ax.set_ylabel("Time (s)")
    ax.set_title("Runtime comparison")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def format_summary_table(summary: Dict[str, Dict[str, object]], metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for method, info in summary.items():
        row = {
            "Method": method,
            "Clusters": info["n_clusters"],
            "Runtime (s)": f"{info['time']:.3f}",
            "Bandwidth": f"{info['bandwidth']:.4f}",
            "Error": f"{info['error']:.4f}",
            "Cluster counts": info.get("counts", "--"),
        }
        for metric_name, value in metrics[method].items():
            row[metric_name] = f"{value:.3f}" if not math.isnan(value) else "--"
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="SAMS Clustering Demo", layout="wide", initial_sidebar_state="expanded")
    st.title("SAMS: Stochastic Approximation Mean-Shift Clustering")
    st.markdown(
        """
        This interactive demo reproduces the core ideas from Hyrien & Baran (2016).
        Generate synthetic datasets, tune SAMS parameters, and compare outcomes with
        deterministic mean-shift baselines.
        """
    )

    with st.sidebar:
        st.header("Dataset")
        dataset_name = st.selectbox(
            "Structure",
            ["Gaussian Blobs", "Mixed Densities"],
        )
        n_features = st.number_input("Dimensions", min_value=1, max_value=20, value=2, step=1)
        n_clusters = st.number_input("Clusters", min_value=2, max_value=20, value=4, step=1)
        n_samples = st.slider("Samples", min_value=500, max_value=20000, value=2000, step=500)
        noise = st.slider("Noise", min_value=0.0, max_value=0.5, value=0.05, step=0.01)
        cluster_std = st.slider("Cluster Std Dev", min_value=0.1, max_value=3.0, value=1.0, step=0.1)

        st.markdown("---")
        st.header("SAMS Parameters")
        bw_mode = st.radio(
            "Bandwidth mode",
            options=["Auto (adaptive)", "Auto (Silverman)", "Manual"],
            index=0,
        )
        if bw_mode == "Manual":
            bandwidth = st.slider("Bandwidth", min_value=0.2, max_value=2.0, value=0.7, step=0.05)
            bandwidth_method = None
        elif bw_mode == "Auto (Silverman)":
            bandwidth = None
            bandwidth_method = "silverman"
        else:
            bandwidth = None
            bandwidth_method = "adaptive"

        sampling_mode = st.radio(
            "Sampling mode",
            options=["Fraction", "Fixed size"],
            index=0,
        )
        if sampling_mode == "Fraction":
            sample_fraction = st.number_input("Sample fraction (ρ)", min_value=0.001, max_value=1.0, value=0.01, step=0.001, format="%.3f")
            fixed_sample_size = None
        else:
            sample_fraction = 1.0  # Will be overridden by fixed_sample_size
            fixed_sample_size = st.number_input("Fixed sample size", min_value=1, max_value=n_samples, value=min(100, n_samples), step=10)

        st.markdown("---")
        st.header("Baselines")
        include_sklearn = st.checkbox("Include sklearn MeanShift", value=True)
        if include_sklearn:
            use_sams_bandwidth = st.checkbox("Use SAMS bandwidth for sklearn", value=False)
        else:
            use_sams_bandwidth = False
        include_custom = st.checkbox("Include custom MeanShift", value=True)

        st.markdown("---")
        random_state = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
        run_button = st.button("Run experiment", type="primary")

    if not run_button:
        st.info("Configure parameters in the sidebar and click **Run experiment**.")
        st.stop()

    cfg = DatasetConfig(
        name=dataset_name,
        n_samples=n_samples,
        n_features=n_features,
        random_state=int(random_state),
        noise=float(noise),
        cluster_std=float(cluster_std),
        n_clusters=int(n_clusters),
    )

    with st.spinner("Generating data and running algorithms…"):
        X, y_true = generate_dataset(cfg)
        summary: Dict[str, Dict[str, object]] = {}
        metrics: Dict[str, Dict[str, float]] = {}

        sams_result = run_sams(
            X,
            y_true,
            bandwidth=bandwidth,
            sample_fraction=sample_fraction,
            seed=int(random_state),
            bandwidth_method=bandwidth_method,
            fixed_sample_size=fixed_sample_size,
        )
        summary[sams_result["label"]] = sams_result
        metrics[sams_result["label"]] = compute_metrics(X, y_true, sams_result["mapped"])

        ms_bandwidth = bandwidth if bandwidth is not None else sams_result["bandwidth"]

        if include_custom:
            labels_custom, runtime_custom, clusters_custom = run_meanshift_custom(X, bandwidth=ms_bandwidth)
            mapped_custom, err_custom = map_labels_to_true(y_true, labels_custom)
            method_name = "MeanShift (custom)"
            summary[method_name] = {
                "labels": labels_custom,
                "mapped": mapped_custom,
                "time": runtime_custom,
                "n_clusters": clusters_custom,
                "bandwidth": ms_bandwidth,
                "error": err_custom,
                "counts": format_cluster_counts(labels_custom),
            }
            metrics[method_name] = compute_metrics(X, y_true, mapped_custom)

        if include_sklearn:
            try:
                sklearn_bw = ms_bandwidth if use_sams_bandwidth else None
                labels_ms, runtime_ms, clusters_ms, actual_bw = run_meanshift_sklearn(X, bandwidth=sklearn_bw)
                mapped_ms, err_ms = map_labels_to_true(y_true, labels_ms)
                method_name = "MeanShift (sklearn)"
                if use_sams_bandwidth:
                    method_name += " [SAMS bw]"
                summary[method_name] = {
                    "labels": labels_ms,
                    "mapped": mapped_ms,
                    "time": runtime_ms,
                    "n_clusters": clusters_ms,
                    "bandwidth": actual_bw,
                    "error": err_ms,
                    "counts": format_cluster_counts(labels_ms),
                }
                metrics[method_name] = compute_metrics(X, y_true, mapped_ms)
            except RuntimeError as exc:
                st.warning(str(exc))

    st.subheader("Clustering visualisations")
    # Add true labels as the first column
    cols = st.columns(len(summary) + 1)
    cols[0].markdown("**True labels**")
    cols[0].pyplot(plot_clusters(X, y_true, "Ground Truth"), clear_figure=True)

    for col, (method, info) in zip(cols[1:], summary.items()):
        col.markdown(f"**{method}**")
        col.pyplot(plot_clusters(X, info["mapped"], method), clear_figure=True)

    result_df = format_summary_table(summary, metrics)
    st.subheader("Performance summary")
    st.dataframe(result_df, use_container_width=True)

    st.subheader("Runtime comparison")
    st.pyplot(plot_runtime(summary), clear_figure=True)

    st.success("Done! Download the summary CSV below if needed.")
    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download summary CSV",
        data=csv_bytes,
        file_name="sams_clustering_summary.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.caption(
        "Demo adapted from https://huggingface.co/spaces/chnrui/sams-clustering-demo using Streamlit."
    )


if __name__ == "__main__":
    main()
