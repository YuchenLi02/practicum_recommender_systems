from __future__ import annotations

import numpy as np


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(lo), float(hi)


def paired_bootstrap_pvalue(
    a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 0
) -> float:
    """Two-sided paired bootstrap p-value for the mean difference a minus b."""
    rng = np.random.default_rng(seed)
    diff = a - b
    n = len(diff)
    if n == 0:
        return float("nan")
    centered = diff - diff.mean()
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = centered[idx].mean(axis=1)
    obs = abs(diff.mean())
    p = (np.abs(boot_means) >= obs).mean()
    return float(p)


def bonferroni(p_values: list[float]) -> list[float]:
    """Bonferroni-correct a list of p-values."""
    m = len(p_values)
    return [min(1.0, p * m) for p in p_values]
