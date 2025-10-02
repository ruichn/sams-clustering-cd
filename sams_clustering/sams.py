"""Implementation of the Stochastic Approximation Mean-Shift (SAMS) algorithm."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Optional

import numpy as np

from .kernels import RadialKernel, ensure_kernel


@dataclass
class SAMSConfig:
    """Hyper-parameters controlling a SAMS run."""

    bandwidth: float | np.ndarray | None = None
    bandwidth_method: str = "adaptive"
    bandwidth_alpha1: float = 1.0
    bandwidth_alpha2: float = 0.5
    bandwidth_sample_size: int = 2048
    bandwidth_min: float = 1e-3
    sample_fraction: float = 0.01
    fixed_sample_size: int | None = None
    max_iter: int = 500
    gamma0: float = 1.0
    gamma_exponent: float = 0.51
    beta0: float = 1.0
    beta_exponent: float = 0.6
    eta_min: float = 1e-3
    eta_max: float = 1e6
    alpha_phi: float = 0.75
    alpha_delta: float = 0.95
    delta_threshold: float = 0.15
    min_iter: int = 5
    merge_radius: Optional[float] = None
    random_state: Optional[int] = None
    kernel: Optional[str] = "gaussian"
    epsilon_sample_size: int = 128
    epsilon_percentile: float = 5.0

    def ensure_valid(self) -> None:
        if self.fixed_sample_size is None and not (0.0 < self.sample_fraction <= 1.0):
            raise ValueError("sample_fraction must lie in (0, 1]")
        if self.fixed_sample_size is not None and self.fixed_sample_size <= 0:
            raise ValueError("fixed_sample_size must be positive when provided")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if not (0.5 < self.gamma_exponent <= 1.0):
            raise ValueError("gamma_exponent must be in (0.5, 1]")
        if not (0.5 < self.beta_exponent <= 1.0):
            raise ValueError("beta_exponent must be in (0.5, 1]")
        if not (0.0 < self.alpha_phi <= 1.0):
            raise ValueError("alpha_phi must be in (0, 1]")
        if not (0.5 < self.alpha_delta <= 1.0):
            raise ValueError("alpha_delta must be in (0.5, 1]")
        if self.delta_threshold < 0.0:
            raise ValueError("delta_threshold must be non-negative")
        if self.merge_radius is not None and self.merge_radius <= 0:
            raise ValueError("merge_radius must be positive when provided")
        if self.epsilon_sample_size <= 0:
            raise ValueError("epsilon_sample_size must be positive")
        if not (0.0 < self.epsilon_percentile <= 100.0):
            raise ValueError("epsilon_percentile must lie in (0, 100]")
        if self.bandwidth_sample_size <= 0:
            raise ValueError("bandwidth_sample_size must be positive")
        if self.bandwidth_alpha1 <= 0:
            raise ValueError("bandwidth_alpha1 must be positive")
        if not (0 <= self.bandwidth_alpha2 <= 1):
            raise ValueError("bandwidth_alpha2 must lie in [0,1]")
        if self.bandwidth_min <= 0:
            raise ValueError("bandwidth_min must be positive")


class SAMS:
    """Stochastic Approximation Mean-Shift clustering."""

    def __init__(self, config: SAMSConfig) -> None:
        config.ensure_valid()
        self.config = config
        self.kernel: RadialKernel = ensure_kernel(config.kernel)
        self.np_rng = np.random.default_rng(config.random_state)
        self._eps: Optional[np.ndarray] = None
        self._last_bandwidths: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        n_samples, _ = X.shape
        if n_samples == 0:
            raise ValueError("X must contain at least one sample")

        bandwidths = self._prepare_bandwidths(X)
        self._eps = self._estimate_epsilons(X, bandwidths)

        trajectories = np.zeros_like(X)
        for i in range(n_samples):
            trajectories[i] = self._run_single_trajectory(X, i, bandwidths)

        merge_radius = self.config.merge_radius
        if merge_radius is None:
            merge_radius = float(np.median(bandwidths)) * 0.5

        modes, labels = self._merge_modes(trajectories, merge_radius)
        return modes, labels

    # ------------------------------------------------------------------
    # Core iteration
    # ------------------------------------------------------------------

    def _run_single_trajectory(self, X: np.ndarray, index: int, bandwidths: np.ndarray) -> np.ndarray:
        cfg = self.config
        n_samples, _ = X.shape
        if cfg.fixed_sample_size is not None:
            m = min(cfg.fixed_sample_size, n_samples)
        else:
            m = max(1, int(math.ceil(cfg.sample_fraction * n_samples)))
            if m > n_samples:
                m = n_samples

        x = X[index].copy()
        eps = self._eps
        assert eps is not None

        b_hat = cfg.eta_min
        b_eta = b_hat
        a_bar = np.zeros_like(x)
        prev_A = None
        s_bar = 0.0

        for k in range(1, cfg.max_iter + 1):
            sample_idx = self._sample_indices(n_samples, m)

            A_curr, B_curr = self._estimate_moments(X, x, bandwidths, sample_idx)

            if k == 1:
                a_bar = A_curr.copy()
                b_hat = B_curr
                b_eta = self._clip_eta(b_hat)
            else:
                phi = 1.0 / (k ** cfg.alpha_phi)
                a_bar = a_bar + phi * (A_curr - a_bar)

                beta_k = cfg.beta0 / (k ** cfg.beta_exponent)
                b_hat = b_hat + beta_k * (B_curr - b_hat)
                b_eta = self._clip_eta(b_hat)

            gamma_k = cfg.gamma0 / (k ** cfg.gamma_exponent)
            if b_eta == 0:
                break
            step = gamma_k * (A_curr / b_eta)
            x_next = x + step

            if prev_A is not None:
                indicator = 1.0 if float(np.dot(A_curr, prev_A)) < 0.0 else 0.0
                delta_k = 1.0 / (k ** cfg.alpha_delta)
                s_bar = s_bar - delta_k * (s_bar - indicator)
            prev_A = A_curr

            if self._should_stop(a_bar, s_bar, k, eps):
                x = x_next
                break

            x = x_next

        return x

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _estimate_moments(
        self,
        X: np.ndarray,
        x: np.ndarray,
        bandwidths: np.ndarray,
        sample_idx: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        if len(sample_idx) == 0:
            return np.zeros_like(x), 0.0
        h = bandwidths[sample_idx]
        diff = X[sample_idx] - x
        scaled = diff / h[:, None]
        norm_sq = np.sum(scaled ** 2, axis=1)
        g_vals = -self.kernel.k_prime(norm_sq)
        weights = (h ** (-(X.shape[1] + 2))) * g_vals
        A = np.sum(weights[:, None] * diff, axis=0) / len(sample_idx)
        B = float(np.sum(weights) / len(sample_idx))
        return A, B

    def _clip_eta(self, value: float) -> float:
        cfg = self.config
        return float(np.clip(value, cfg.eta_min, cfg.eta_max))

    def _should_stop(self, a_bar: np.ndarray, s_bar: float, k: int, eps: np.ndarray) -> bool:
        cfg = self.config
        if k < cfg.min_iter:
            return False

        if np.all(np.abs(a_bar) <= eps):
            return True

        lower_bound = s_bar - self._z_value(0.95) / (2.0 * (k ** (cfg.alpha_delta / 2.0)))
        if lower_bound > 0.5 - cfg.delta_threshold:
            return True
        return False

    def _estimate_epsilons(self, X: np.ndarray, bandwidths: np.ndarray) -> np.ndarray:
        cfg = self.config
        n_samples, _ = X.shape
        sample_size = min(cfg.epsilon_sample_size, n_samples)
        point_indices = self.np_rng.choice(n_samples, size=sample_size, replace=False)
        support_indices = self.np_rng.choice(n_samples, size=sample_size, replace=False)

        gradient_samples = []
        for idx in point_indices:
            grad, _ = self._estimate_moments(X, X[idx], bandwidths, support_indices)
            gradient_samples.append(np.abs(grad))

        if not gradient_samples:
            return np.full(X.shape[1], 1e-3, dtype=float)

        gradient_arr = np.vstack(gradient_samples)
        epsilons = np.percentile(gradient_arr, cfg.epsilon_percentile, axis=0)
        if np.any(epsilons <= 0):
            positive = epsilons[epsilons > 0]
            replacement = positive.min() if positive.size else 1e-6
            epsilons[epsilons <= 0] = replacement
        return epsilons

    def _prepare_bandwidths(self, X: np.ndarray) -> np.ndarray:
        cfg = self.config
        bw = cfg.bandwidth
        n_samples = X.shape[0]
        if bw is None:
            method = cfg.bandwidth_method.lower()
            if method == "adaptive":
                result = self._adaptive_bandwidths(X)
                self._last_bandwidths = result.copy()
                return result
            elif method == "silverman":
                scalar = self._silverman_bandwidth(X)
                scalar = max(scalar, cfg.bandwidth_min)
                result = np.full(n_samples, scalar, dtype=float)
                self._last_bandwidths = result.copy()
                return result
            else:
                raise ValueError(f"Unknown bandwidth_method '{cfg.bandwidth_method}'")
        if np.isscalar(bw):
            if float(bw) <= 0:
                raise ValueError("bandwidth must be positive")
            result = np.full(n_samples, float(bw), dtype=float)
            self._last_bandwidths = result.copy()
            return result
        bw_array = np.asarray(bw, dtype=float)
        if bw_array.shape not in {(n_samples,), (n_samples, 1)}:
            raise ValueError("bandwidth array must have shape (n_samples,) or be scalar")
        bw_array = bw_array.reshape(n_samples)
        if np.any(bw_array <= 0):
            raise ValueError("bandwidth entries must be positive")
        self._last_bandwidths = bw_array.copy()
        return bw_array

    def _sample_indices(self, n: int, m: int) -> np.ndarray:
        if m >= n:
            return np.arange(n)
        return self.np_rng.choice(n, size=m, replace=False)

    def _silverman_bandwidth(self, X: np.ndarray) -> float:
        n, d = X.shape
        std = np.std(X, axis=0, ddof=1)
        sigma = np.mean(std)
        if sigma <= 0:
            sigma = 1.0
        factor = (4 / (d + 2)) ** (1 / (d + 4))
        return factor * sigma * (n ** (-1 / (d + 4)))

    def _adaptive_bandwidths(self, X: np.ndarray) -> np.ndarray:
        cfg = self.config
        n_samples, n_features = X.shape
        support_size = min(cfg.bandwidth_sample_size, n_samples)
        support_idx = self.np_rng.choice(n_samples, size=support_size, replace=False)
        support = X[support_idx]

        h0 = self._silverman_bandwidth(support)
        h0 = max(h0, cfg.bandwidth_min)

        diff = X[:, None, :] - support[None, :, :]
        scaled = diff / h0
        norm_sq = np.sum(scaled ** 2, axis=2)
        kernel_vals = np.exp(-0.5 * norm_sq)
        norm_const = (2 * np.pi) ** (-n_features / 2) * (1 / (h0 ** n_features))
        f_tilde = norm_const * kernel_vals.mean(axis=1)

        eps = 1e-12
        f_tilde = np.clip(f_tilde, eps, None)
        beta_hat = np.exp(np.mean(np.log(f_tilde)))

        hi = cfg.bandwidth_alpha1 * h0 * (beta_hat / f_tilde) ** cfg.bandwidth_alpha2
        hi = np.clip(hi, cfg.bandwidth_min, None)
        return hi.astype(float)

    def _merge_modes(self, points: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
        centers: list[np.ndarray] = []
        counts: list[int] = []
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

        modes = np.vstack(centers)
        return modes, labels

    @staticmethod
    def _z_value(quantile: float) -> float:
        return NormalDist().inv_cdf(quantile)
