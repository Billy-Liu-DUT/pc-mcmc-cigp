from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from pc_mcmc_cigp.discovery.diagnostics import effective_sample_size, gelman_rubin_rhat
from pc_mcmc_cigp.discovery.engine import MechanismEngine
from pc_mcmc_cigp.discovery.sampler import DiscoveryResult, MCMCConfig, SpikeAndSlabSampler


def run_multiple_chains(
    engine: MechanismEngine,
    dataset: Sequence[dict],
    config: MCMCConfig | None = None,
    n_chains: int = 4,
    candidate_pathways: Sequence[Sequence[int]] | None = None,
) -> DiscoveryResult:
    """Run independent MCMC chains and merge posterior inclusion probabilities."""

    if n_chains < 1:
        raise ValueError("n_chains must be >= 1")
    base_config = config or MCMCConfig()
    base_seed = base_config.random_state if base_config.random_state is not None else 0
    results = []
    z_chains = []
    theta_chains = []

    for chain_idx in range(n_chains):
        chain_config = replace(base_config, random_state=base_seed + chain_idx)
        sampler = SpikeAndSlabSampler(engine, chain_config)
        results.append(sampler.fit(dataset, candidate_pathways=candidate_pathways))
        z_chains.append(sampler.chain_z_array)
        theta_chains.append(sampler.chain_theta_array)

    z_stack = _trim_and_stack(z_chains)
    theta_stack = _trim_and_stack(theta_chains)
    pip = np.mean(z_stack, axis=(0, 1))
    mean_parameters = np.mean([result.mean_parameters for result in results], axis=0)
    rhat = gelman_rubin_rhat(z_stack)
    ess = effective_sample_size(z_stack)
    best_result = max(results, key=lambda result: result.diagnostics.get("map_log_posterior", -np.inf))
    selected = [
        reaction
        for reaction, probability in zip(engine.reactions, pip)
        if probability >= base_config.selection_threshold
    ]
    if not selected and engine.reactions:
        selected = [engine.reactions[int(np.argmax(pip))]]
    diagnostics = {
        "n_chains": n_chains,
        "n_samples": int(z_stack.shape[0] * z_stack.shape[1]),
        "acceptance_rate": float(np.mean([r.diagnostics["acceptance_rate"] for r in results])),
        "n_active_mean": float(np.mean(np.sum(z_stack, axis=2))),
        "n_active_std": float(np.std(np.sum(z_stack, axis=2))),
        "map_log_posterior": float(best_result.diagnostics["map_log_posterior"]),
        "rhat_max": float(np.nanmax(rhat)),
        "ess_min": float(np.nanmin(ess)),
        "map_rmse": float(best_result.diagnostics.get("map_rmse", np.nan)),
        "invalid_evaluations": int(sum(r.diagnostics.get("invalid_evaluations", 0) for r in results)),
        "n_candidate_pathways": len(candidate_pathways or []),
    }
    selected_pathways = None
    pathway_pip = np.empty(0, dtype=float)
    if candidate_pathways:
        pathway_pip = np.asarray([
            np.mean(np.prod(z_stack[:, :, np.asarray(path, dtype=int)], axis=2))
            for path in candidate_pathways
        ])
        selected_pathways = [
            tuple(path) for path, probability in zip(candidate_pathways, pathway_pip)
            if probability >= base_config.selection_threshold
        ]
    return DiscoveryResult(
        posterior_inclusion_probabilities=pip,
        mean_parameters=mean_parameters,
        selected_reactions=selected,
        diagnostics=diagnostics,
        map_structure=best_result.map_structure,
        map_parameters=best_result.map_parameters,
        chain_diagnostics={
            "rhat": rhat, "ess": ess, "pathway_pip": pathway_pip,
            "parameter_q05": np.quantile(theta_stack, 0.05, axis=(0, 1)),
            "parameter_q95": np.quantile(theta_stack, 0.95, axis=(0, 1)),
        },
        selected_pathways=selected_pathways,
    )


def _trim_and_stack(chains: list[np.ndarray]) -> np.ndarray:
    min_len = min(chain.shape[0] for chain in chains)
    return np.stack([chain[:min_len] for chain in chains], axis=0)
