"""Simulation study comparing SAMS sampled fractions.

This script reproduces (at a smaller scale) the simulation study described in
Hyrien & Baran (2016). It contrasts the reference mean-shift trajectory
(sample_fraction = 1.0) with stochastic subsampling settings and reports the
mis-clustering rate relative to the reference along with runtimes. Two plots are
written to ``experiments/output`` summarising the findings.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from sklearn.cluster import MeanShift, estimate_bandwidth
except ImportError:  # pragma: no cover - optional dependency
    MeanShift = None
    estimate_bandwidth = None
import numpy as np

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from sams_clustering import SAMS, SAMSConfig


@dataclass
class SimulationResult:
    method: str
    sample_fraction: float | None
    mean_error: float
    std_error: float
    mean_runtime: float
    std_runtime: float
    mean_clusters: float
    sample_points: np.ndarray | None = None
    sample_labels: np.ndarray | None = None
    true_labels: np.ndarray | None = None


def merge_points(points: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray]:
    centers: List[np.ndarray] = []
    counts: List[int] = []
    labels = np.empty(len(points), dtype=int)

    for i, point in enumerate(points):
        assigned = False
        for idx, center in enumerate(centers):
            if np.linalg.norm(point - center) <= radius:
                counts[idx] += 1
                centers[idx] = center + (point - center) / counts[idx]
                labels[i] = idx
                assigned = True
                break
        if not assigned:
            centers.append(point.copy())
            counts.append(1)
            labels[i] = len(centers) - 1

    return np.vstack(centers) if centers else np.empty((0, points.shape[1])), labels


def map_labels_to_true(true_labels: np.ndarray, pred_labels: np.ndarray) -> Tuple[np.ndarray, float]:
    mapping: Dict[int, int] = {}
    for label in np.unique(pred_labels):
        mask = pred_labels == label
        if not np.any(mask):
            continue
        values = true_labels[mask]
        if values.size == 0:
            mapping[label] = -1
            continue
        counts = Counter(values)
        mapped_label, _ = counts.most_common(1)[0]
        mapping[label] = mapped_label

    mapped = np.vectorize(lambda x: mapping.get(x, -1))(pred_labels)
    error = float(np.mean(mapped != true_labels))
    return mapped, error


def run_meanshift_sklearn(
    X: np.ndarray,
    bandwidth: float | None = None,
    merge_radius: float = 0.6,
) -> Tuple[np.ndarray, float, int, float]:
    if MeanShift is None:
        raise RuntimeError(
            "scikit-learn is required for MeanShift. Install it via 'pip install scikit-learn'."
        )

    if bandwidth is None and estimate_bandwidth is not None:
        bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=min(1000, len(X)))
    if bandwidth is None or bandwidth <= 0:
        bandwidth = 0.7

    start = time.perf_counter()
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    runtime = time.perf_counter() - start
    labels = ms.labels_
    n_clusters = len(np.unique(labels))
    return labels, runtime, n_clusters, bandwidth


def run_meanshift_custom(
    X: np.ndarray,
    bandwidth: float = 0.7,
    merge_radius: float = 0.6,
    max_iter: int = 100,
    tol: float = 1e-3,
) -> Tuple[np.ndarray, float, int]:
    n_samples, _ = X.shape
    trajectories = np.zeros_like(X)

    start = time.perf_counter()
    for i, x0 in enumerate(X):
        y = x0.copy()
        for _ in range(max_iter):
            diff = X - y
            r2 = np.sum((diff / bandwidth) ** 2, axis=1)
            weights = np.exp(-0.5 * r2)
            denom = np.sum(weights)
            if denom <= 1e-12:
                break
            y_new = np.sum(weights[:, None] * X, axis=0) / denom
            if np.linalg.norm(y_new - y) < tol * bandwidth:
                y = y_new
                break
            y = y_new
        trajectories[i] = y

    modes, labels = merge_points(trajectories, merge_radius)
    runtime = time.perf_counter() - start
    n_clusters = len(np.unique(labels))
    return labels, runtime, n_clusters


def generate_dataset(n_samples: int = 8000, random_state: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    # Mixture components roughly matching the 2D example in the paper
    means = np.array([
        [0.0, 0.0],
        [3.5, 0.0],
        [1.75, 3.0],
    ])
    cov = np.array([[0.18, 0.03], [0.03, 0.22]])
    choices = rng.choice(len(means), size=n_samples, p=[0.35, 0.35, 0.30])
    data = np.empty((n_samples, 2))
    for k, mean in enumerate(means):
        mask = choices == k
        count = mask.sum()
        if count:
            data[mask] = rng.multivariate_normal(mean, cov, size=count, check_valid="warn")
    # Add a touch of uniform noise
    noise_mask = rng.random(n_samples) < 0.01
    data[noise_mask] += rng.uniform(-0.6, 0.6, size=(noise_mask.sum(), 2))
    labels = choices.astype(int)
    labels[noise_mask] = -1
    return data, labels


def run_sams_single(
    X: np.ndarray,
    y_true: np.ndarray,
    sample_fraction: float,
    seed: int,
) -> Tuple[float, float, int, np.ndarray]:
    config = SAMSConfig(
        bandwidth=0.7,
        sample_fraction=sample_fraction,
        max_iter=300,
        random_state=seed,
        merge_radius=0.6,
    )
    model = SAMS(config)
    start = time.perf_counter()
    modes, labels = model.fit(X)
    runtime = time.perf_counter() - start
    mapped_labels, error_rate = map_labels_to_true(y_true, labels)
    return error_rate, runtime, len(np.unique(labels)), mapped_labels


def run_simulation(
    sample_fractions: Iterable[float],
    n_repeats: int = 5,
    random_state: int = 0,
    n_samples: int = 8000,
    *,
    use_sklearn: bool = True,
    verbose: bool = True,
) -> List[SimulationResult]:
    X, y_true = generate_dataset(n_samples=n_samples, random_state=random_state)

    results: List[SimulationResult] = []
    for frac in sample_fractions:
        errors = []
        runtimes = []
        clusters = []
        sample_labels = None
        for rep in range(n_repeats):
            error, runtime, n_clusters, mapped_labels = run_sams_single(
                X,
                y_true,
                sample_fraction=frac,
                seed=random_state + rep + 1,
            )
            errors.append(error)
            runtimes.append(runtime)
            clusters.append(n_clusters)
            if sample_labels is None:
                sample_labels = mapped_labels
        results.append(
            SimulationResult(
                method="SAMS",
                sample_fraction=frac,
                mean_error=float(np.mean(errors)),
                std_error=float(np.std(errors, ddof=1) if len(errors) > 1 else 0.0),
                mean_runtime=float(np.mean(runtimes)),
                std_runtime=float(np.std(runtimes, ddof=1) if len(runtimes) > 1 else 0.0),
                mean_clusters=float(np.mean(clusters)),
                sample_points=X,
                sample_labels=sample_labels,
                true_labels=y_true,
            )
        )

    # scikit-learn mean shift
    if use_sklearn:
        try:
            labels_sklearn, runtime_sklearn, clusters_sklearn = run_meanshift_sklearn(X)
            mapped_sklearn, err_sklearn = map_labels_to_true(y_true, labels_sklearn)
            results.append(
                SimulationResult(
                    method="MS-sklearn",
                    sample_fraction=None,
                    mean_error=err_sklearn,
                    std_error=0.0,
                    mean_runtime=runtime_sklearn,
                    std_runtime=0.0,
                    mean_clusters=float(clusters_sklearn),
                    sample_points=X,
                    sample_labels=mapped_sklearn,
                    true_labels=y_true,
                )
            )
        except RuntimeError as exc:
            if verbose:
                print(f"Skipping sklearn MeanShift: {exc}")

    # custom mean shift
    labels_custom, runtime_custom, clusters_custom = run_meanshift_custom(X)
    mapped_custom, err_custom = map_labels_to_true(y_true, labels_custom)
    results.append(
        SimulationResult(
            method="MS-custom",
            sample_fraction=None,
            mean_error=err_custom,
            std_error=0.0,
            mean_runtime=runtime_custom,
            std_runtime=0.0,
            mean_clusters=float(clusters_custom),
            sample_points=X,
            sample_labels=mapped_custom,
            true_labels=y_true,
        )
    )

    return results


def format_table(results: List[SimulationResult]) -> str:
    headers = (
        "method",
        "rho",
        "error_mean",
        "error_std",
        "time_mean(s)",
        "time_std(s)",
        "clusters",
    )
    rows = [headers]
    for res in results:
        rho_display = f"{res.sample_fraction:.4f}" if res.sample_fraction is not None else "--"
        rows.append(
            (
                res.method,
                rho_display,
                f"{res.mean_error:.4f}",
                f"{res.std_error:.4f}",
                f"{res.mean_runtime:.3f}",
                f"{res.std_runtime:.3f}",
                f"{res.mean_clusters:.1f}",
            )
        )
    col_widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    lines = []
    for idx, row in enumerate(rows):
        line = " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))
        lines.append(line)
        if idx == 0:
            lines.append("-+-".join("-" * w for w in col_widths))
    return "\n".join(lines)


def results_to_records(results: List[SimulationResult]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for res in results:
        records.append(
            {
                "method": res.method,
                "rho": float(res.sample_fraction) if res.sample_fraction is not None else None,
                "error_mean": res.mean_error,
                "error_std": res.std_error,
                "time_mean_s": res.mean_runtime,
                "time_std_s": res.std_runtime,
                "clusters": res.mean_clusters,
            }
        )
    return records


def create_figures(results: List[SimulationResult]) -> Tuple[Optional[plt.Figure], Optional[plt.Figure], Optional[plt.Figure]]:
    sams_results = sorted(
        (res for res in results if res.method == "SAMS"),
        key=lambda r: r.sample_fraction if r.sample_fraction is not None else float("inf"),
    )
    other_results = [res for res in results if res.method != "SAMS"]

    fig_perf, fig_clusters, fig_scatter = None, None, None

    if sams_results or other_results:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        if sams_results:
            fractions = np.array([res.sample_fraction for res in sams_results], dtype=float)
            errors = np.array([res.mean_error for res in sams_results])
            runtimes = np.array([res.mean_runtime for res in sams_results])

            axes[0].plot(fractions, errors, marker="o", label="SAMS")
            axes[1].plot(fractions, runtimes, marker="o", color="tab:orange", label="SAMS")

            for ax in axes:
                ax.set_xscale("log")
                ax.invert_xaxis()
                ax.set_xlabel(r"Sample fraction $\rho$")
        else:
            axes[0].set_xlabel("Method")
            axes[1].set_xlabel("Method")

        axes[0].set_ylabel("Mis-clustering rate")
        axes[0].set_title("Accuracy vs. sample fraction")
        axes[1].set_ylabel("Runtime (s)")
        axes[1].set_title("Runtime vs. sample fraction")

        for res in other_results:
            axes[0].axhline(res.mean_error, linestyle="--", label=res.method)
            axes[1].axhline(res.mean_runtime, linestyle="--", label=res.method)

        axes[0].legend(fontsize="small")
        axes[1].legend(fontsize="small")
        fig.tight_layout()
        fig_perf = fig

    methods = []
    method_values = []
    for res in sams_results:
        label = f"SAMS (ρ={res.sample_fraction:.3f})" if res.sample_fraction is not None else "SAMS"
        methods.append(label)
        method_values.append(res.mean_clusters)
    for res in other_results:
        methods.append(res.method)
        method_values.append(res.mean_clusters)

    if methods:
        fig2, ax2 = plt.subplots(figsize=(max(5, len(methods) * 1.2), 4))
        ax2.bar(range(len(methods)), method_values, color="tab:green")
        ax2.set_xticks(range(len(methods)))
        ax2.set_xticklabels(methods, rotation=30, ha="right")
        ax2.set_ylabel("Average clusters")
        ax2.set_title("Clusters detected per method")
        fig2.tight_layout()
        fig_clusters = fig2

    pts = results[0].sample_points if results else None
    true_labels = results[0].true_labels if results else None
    scatter_entries: List[Tuple[str, np.ndarray]] = []
    if pts is not None and true_labels is not None:
        scatter_entries.append(("Ground truth", true_labels))
        if sams_results:
            best_sams = min(sams_results, key=lambda r: r.sample_fraction or 1.0)
            if best_sams.sample_labels is not None:
                scatter_entries.append((f"SAMS (ρ={best_sams.sample_fraction:g})", best_sams.sample_labels))
        for res in other_results:
            if res.sample_labels is not None:
                scatter_entries.append((res.method, res.sample_labels))

    if pts is not None and scatter_entries:
        fig3, axes = plt.subplots(1, len(scatter_entries), figsize=(5 * len(scatter_entries), 5), sharex=True, sharey=True)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        def _scatter(ax: plt.Axes, labels: np.ndarray, title: str) -> None:
            unique_labels = np.unique(labels)
            cmap = plt.colormaps["tab20"].resampled(max(len(unique_labels), 1))
            for idx, label in enumerate(unique_labels):
                mask = labels == label
                color = cmap(idx)
                label_name = "noise" if label == -1 else str(label)
                ax.scatter(pts[mask, 0], pts[mask, 1], s=6, color=color, label=label_name, alpha=0.7)
            ax.set_title(title)
            ax.set_xlabel("x₁")
            ax.set_ylabel("x₂")
            ax.legend(markerscale=2, fontsize="small", loc="upper right")
            ax.grid(True, linestyle="--", alpha=0.3)

        for ax, (title, labels) in zip(axes, scatter_entries):
            _scatter(ax, labels, title)

        fig3.tight_layout()
        fig_scatter = fig3

    return fig_perf, fig_clusters, fig_scatter
def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAMS simulation study")
    parser.add_argument("--repeats", type=int, default=5, help="Number of repeats per fraction")
    parser.add_argument("--samples", type=int, default=8000, help="Number of observations in the synthetic dataset")
    parser.add_argument(
        "--fractions",
        nargs="*",
        type=float,
        default=[1.0, 0.02, 0.01, 0.005],
        help="Sample fractions (rho) to evaluate",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--outdir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent / "output",
        help="Directory to store generated plots",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    results = run_simulation(
        args.fractions,
        n_repeats=args.repeats,
        random_state=args.seed,
        n_samples=args.samples,
    )
    print(format_table(results))

    fig_perf, fig_clusters, fig_scatter = create_figures(results)

    saved_paths: List[str] = []
    if fig_perf is not None:
        runtime_path = args.outdir / "performance.png"
        fig_perf.savefig(runtime_path, dpi=200)
        plt.close(fig_perf)
        saved_paths.append(str(runtime_path))
    if fig_clusters is not None:
        cluster_path = args.outdir / "clusters.png"
        fig_clusters.savefig(cluster_path, dpi=200)
        plt.close(fig_clusters)
        saved_paths.append(str(cluster_path))
    if fig_scatter is not None:
        scatter_path = args.outdir / "clusters_scatter.png"
        fig_scatter.savefig(scatter_path, dpi=200)
        plt.close(fig_scatter)
        saved_paths.append(str(scatter_path))

    if saved_paths:
        print("Plots saved to " + ", ".join(saved_paths))


if __name__ == "__main__":
    main()
