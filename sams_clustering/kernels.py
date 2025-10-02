"""Kernel profiles used by SAMS."""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np


class RadialKernel(Protocol):
    """Interface for radially symmetric kernel profiles."""

    def k(self, r2: np.ndarray) -> np.ndarray:
        """Kernel profile K(||x||^2)."""

    def k_prime(self, r2: np.ndarray) -> np.ndarray:
        """Derivative K'(||x||^2) with respect to the squared radius."""


class GaussianKernel:
    """Gaussian kernel profile K(u) = exp(-u / 2)."""

    def k(self, r2: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * r2)

    def k_prime(self, r2: np.ndarray) -> np.ndarray:
        # d/du exp(-u/2) = -0.5 exp(-u/2)
        return -0.5 * self.k(r2)


DEFAULT_KERNEL = GaussianKernel()


def ensure_kernel(name: str | None) -> RadialKernel:
    if name is None or name.lower() == "gaussian":
        return DEFAULT_KERNEL
    raise ValueError(f"Unsupported kernel '{name}'")
