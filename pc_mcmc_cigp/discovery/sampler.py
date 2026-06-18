from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from pc_mcmc_cigp.discovery.engine import MechanismEngine
from pc_mcmc_cigp.reactions import Reaction


@dataclass(frozen=True)
class MCMCConfig:
    n_steps: int = 10_000
    burn_in: int = 2_000
    prior_sparsity: float = 0.01
    sigma_likelihood: float = 0.05
    enable_thermo_constraints: bool = True
    isothermal_mode: bool = True
    mu_bounds: tuple[float, float] = (-3.0, 3.0)
    ln_k_bounds: tuple[float, float] = (-1.0, 5.0)
    k_bounds: tuple[float, float] = (0.01, 100.0)
    step_size: float = 0.25
    random_state: int | None = None
    selection_threshold: float = 0.5


@dataclass(frozen=True)
class DiscoveryResult:
    posterior_inclusion_probabilities: np.ndarray
    mean_parameters: np.ndarray
    selected_reactions: list[Reaction]
    diagnostics: dict[str, float | int]
    map_structure: np.ndarray
    map_parameters: np.ndarray
    chain_diagnostics: dict[str, np.ndarray] | None = None


class SpikeAndSlabSampler:
    """Spike-and-slab Metropolis sampler for sparse reaction topology inference."""

    def __init__(self, engine: MechanismEngine, config: MCMCConfig | None = None) -> None:
        self.engine = engine
        self.config = config or MCMCConfig()
        if self.config.burn_in >= self.config.n_steps:
            raise ValueError("burn_in must be smaller than n_steps")
        self.rng = np.random.default_rng(self.config.random_state)
        self.chain_z: list[np.ndarray] = []
        self.chain_theta: list[np.ndarray] = []
        self.chain_z_array = np.empty((0, self.engine.n_reactions))
        self.chain_theta_array = np.empty((0, self._n_params()))

    def fit(self, dataset: Sequence[dict]) -> DiscoveryResult:
        n_params = self._n_params()
        theta = self._initial_theta(n_params)
        z = np.zeros(self.engine.n_reactions, dtype=float)
        if self.engine.n_reactions:
            z[self.rng.integers(0, self.engine.n_reactions)] = 1.0
        score = self._log_posterior(theta, z, dataset)
        accepted = 0
        map_score = -np.inf
        map_z = z.copy()
        map_theta = theta.copy()

        for step in range(self.config.n_steps):
            proposal_theta = self._propose_theta(theta)
            proposal_score = self._log_posterior(proposal_theta, z, dataset)
            if self._accept(proposal_score - score):
                theta = proposal_theta
                score = proposal_score
                accepted += 1

            if self.engine.n_reactions:
                proposal_z = z.copy()
                flip = self.rng.integers(0, self.engine.n_reactions)
                proposal_z[flip] = 1.0 - proposal_z[flip]
                proposal_score = self._log_posterior(theta, proposal_z, dataset)
                if self._accept(proposal_score - score):
                    z = proposal_z
                    score = proposal_score
                    accepted += 1

            if step >= self.config.burn_in:
                self.chain_z.append(z.copy())
                self.chain_theta.append(theta.copy())
                if score > map_score:
                    map_score = score
                    map_z = z.copy()
                    map_theta = theta.copy()

        pip = np.mean(np.asarray(self.chain_z), axis=0)
        chain_theta = np.asarray(self.chain_theta)
        chain_z = np.asarray(self.chain_z)
        self.chain_z_array = chain_z
        self.chain_theta_array = chain_theta
        mean_theta = np.mean(chain_theta, axis=0)
        mean_rates = self._theta_to_rates(mean_theta)
        map_rates = self._theta_to_rates(map_theta)
        selected = [
            reaction for reaction, prob in zip(self.engine.reactions, pip) if prob >= self.config.selection_threshold
        ]
        if not selected and self.engine.reactions:
            selected = [self.engine.reactions[int(np.argmax(pip))]]
        return DiscoveryResult(
            posterior_inclusion_probabilities=pip,
            mean_parameters=mean_rates,
            selected_reactions=selected,
            diagnostics={
                "n_samples": len(self.chain_z),
                "acceptance_rate": accepted / max(1, 2 * self.config.n_steps),
                "n_active_mean": float(np.mean(np.sum(chain_z, axis=1))),
                "n_active_std": float(np.std(np.sum(chain_z, axis=1))),
                "map_log_posterior": float(map_score),
            },
            map_structure=map_z,
            map_parameters=map_rates,
            chain_diagnostics=None,
        )

    def _n_params(self) -> int:
        if self.config.enable_thermo_constraints and self.config.isothermal_mode:
            return self.engine.n_species + self.engine.n_reactions
        return self.engine.n_reactions

    def _initial_theta(self, n_params: int) -> np.ndarray:
        if self.config.enable_thermo_constraints and self.config.isothermal_mode:
            mu = self.rng.uniform(*self.config.mu_bounds, size=self.engine.n_species)
            ln_k = self.rng.uniform(*self.config.ln_k_bounds, size=self.engine.n_reactions)
            return np.concatenate([mu, ln_k])
        lb, ub = self.config.k_bounds
        return self.rng.uniform(np.log(lb), np.log(ub), size=n_params)

    def _propose_theta(self, theta: np.ndarray) -> np.ndarray:
        proposal = theta + self.rng.normal(0.0, self.config.step_size, size=theta.shape)
        if self.config.enable_thermo_constraints and self.config.isothermal_mode:
            proposal[: self.engine.n_species] = np.clip(
                proposal[: self.engine.n_species], *self.config.mu_bounds
            )
            proposal[self.engine.n_species :] = np.clip(
                proposal[self.engine.n_species :], *self.config.ln_k_bounds
            )
        return proposal

    def _theta_to_rates(self, theta: np.ndarray) -> np.ndarray:
        if self.config.enable_thermo_constraints and self.config.isothermal_mode:
            mu = {s.name: theta[i] for i, s in enumerate(self.engine.species)}
            return self.engine.calculate_isothermal_rates(mu, theta[self.engine.n_species :])
        return np.exp(theta)

    def _log_posterior(self, theta: np.ndarray, z: np.ndarray, dataset: Sequence[dict]) -> float:
        rates = self._theta_to_rates(theta)
        try:
            loss = self.engine.calculate_loss(rates, z, dataset)
        except (RuntimeError, FloatingPointError, ValueError):
            return -1e100
        if not np.isfinite(loss):
            return -1e100
        p = np.clip(self.config.prior_sparsity, 1e-9, 1 - 1e-9)
        log_prior = np.sum(z) * np.log(p) + (len(z) - np.sum(z)) * np.log(1 - p)
        return -0.5 * loss / (self.config.sigma_likelihood**2) + float(log_prior)

    def _accept(self, log_ratio: float) -> bool:
        return bool(np.log(self.rng.random()) < min(0.0, log_ratio))
