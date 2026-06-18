from __future__ import annotations

import numpy as np


def gelman_rubin_rhat(chains: np.ndarray) -> np.ndarray:
    """Compute split-free Gelman-Rubin R-hat for shape (chains, draws, dims)."""

    chains = _as_chain_array(chains)
    m, n, _ = chains.shape
    if m < 2 or n < 2:
        return np.full(chains.shape[2], np.nan)
    chain_means = np.mean(chains, axis=1)
    chain_vars = np.var(chains, axis=1, ddof=1)
    within = np.mean(chain_vars, axis=0)
    between = n * np.var(chain_means, axis=0, ddof=1)
    var_hat = ((n - 1) / n) * within + between / n
    return np.sqrt(var_hat / np.maximum(within, 1e-12))


def effective_sample_size(chains: np.ndarray) -> np.ndarray:
    """Conservative ESS estimate using lag-1 autocorrelation per dimension."""

    chains = _as_chain_array(chains)
    m, n, d = chains.shape
    flat_n = m * n
    ess = np.zeros(d, dtype=float)
    for dim in range(d):
        values = chains[:, :, dim]
        autocorrs = []
        for chain in values:
            if len(chain) < 3 or np.var(chain) < 1e-12:
                autocorrs.append(0.0)
                continue
            x0 = chain[:-1] - np.mean(chain[:-1])
            x1 = chain[1:] - np.mean(chain[1:])
            denom = np.sqrt(np.sum(x0**2) * np.sum(x1**2))
            autocorrs.append(float(np.sum(x0 * x1) / denom) if denom > 0 else 0.0)
        rho = float(np.clip(np.mean(autocorrs), -0.99, 0.99))
        ess[dim] = flat_n * (1 - rho) / (1 + rho)
    return np.maximum(ess, 1.0)


def _as_chain_array(chains: np.ndarray) -> np.ndarray:
    arr = np.asarray(chains, dtype=float)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError("chains must have shape (chains, draws, dims)")
    return arr
