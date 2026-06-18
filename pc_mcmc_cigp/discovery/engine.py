from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np
try:
    from scipy.integrate import solve_ivp
except ImportError:  # pragma: no cover - exercised in minimal environments
    solve_ivp = None

from pc_mcmc_cigp.reactions import Reaction, Species


class MechanismEngine:
    """Mass-action ODE engine for candidate reaction networks."""

    def __init__(
        self,
        species: Sequence[Species],
        reactions: Sequence[Reaction],
        stiffness_cap_k: float = 1e6,
        solver_backend: str = "auto",
    ) -> None:
        self.species = list(species)
        self.reactions = list(reactions)
        self.n_species = len(self.species)
        self.n_reactions = len(self.reactions)
        self.stiffness_cap_k = stiffness_cap_k
        self.solver_backend = solver_backend
        self.last_solver_backend = "uninitialized"
        self.s_map = {species.name: i for i, species in enumerate(self.species)}
        self.S = np.column_stack(
            [reaction.stoichiometric_vector(self.species) for reaction in self.reactions]
        ) if self.reactions else np.zeros((self.n_species, 0))
        self.rate_patterns = [self._rate_pattern(reaction) for reaction in self.reactions]
        self.reverse_map = self._detect_reversible_pairs()

    def calculate_isothermal_rates(self, mu_map: dict[str, float], ln_k_forward: Sequence[float]) -> np.ndarray:
        """Convert chemical potentials and forward log rates into detailed-balance rates."""

        ln_k_forward = np.asarray(ln_k_forward, dtype=float)
        k = np.zeros(self.n_reactions, dtype=float)
        primary_forward: dict[int, float] = {}

        for i, reaction in enumerate(self.reactions):
            delta_mu = sum(mu_map.get(s.name, 0.0) for s in reaction.products) - sum(
                mu_map.get(s.name, 0.0) for s in reaction.reactants
            )
            if i in self.reverse_map:
                primary = self.reverse_map[i]
                k_val = primary_forward.get(primary, k[primary]) * np.exp(delta_mu)
            else:
                k_val = np.exp(ln_k_forward[i])
                primary_forward[i] = k_val
            k[i] = float(np.clip(k_val, 1e-12, self.stiffness_cap_k))
        return k

    def simulate(
        self,
        k_vector: Sequence[float],
        z_structure: Sequence[float],
        y0: Sequence[float],
        t_eval: Sequence[float],
        method: str = "LSODA",
        rtol: float = 1e-6,
        atol: float = 1e-9,
    ) -> np.ndarray:
        t_eval = np.asarray(t_eval, dtype=float)
        y0 = np.asarray(y0, dtype=float)
        k_arr = np.asarray(k_vector, dtype=float)
        z_arr = np.asarray(z_structure, dtype=float)
        if self.solver_backend in {"auto", "scipy"} and solve_ivp is not None:
            try:
                solution = solve_ivp(
                    lambda t, y: self._ode(t, y, k_arr, z_arr),
                    (float(t_eval[0]), float(t_eval[-1])),
                    y0,
                    t_eval=t_eval,
                    method=method,
                    rtol=rtol,
                    atol=atol,
                )
                if solution.success:
                    self.last_solver_backend = "scipy"
                    return np.maximum(solution.y, 0.0)
                if self.solver_backend == "scipy":
                    raise RuntimeError(f"ODE integration failed: {solution.message}")
            except Exception:
                if self.solver_backend == "scipy":
                    raise
        self.last_solver_backend = "rk4"
        return self._integrate_rk4(k_arr, z_arr, y0, t_eval)

    def calculate_loss(self, k_vector: Sequence[float], z_structure: Sequence[float], dataset: Sequence[dict]) -> float:
        total = 0.0
        for experiment in dataset:
            pred = self.simulate(k_vector, z_structure, experiment["y0_full"], experiment["t"])
            obs_indices = experiment.get("obs_indices", list(range(self.n_species)))
            residual = pred[np.asarray(obs_indices), :] - np.asarray(experiment["data_matrix"], dtype=float)
            total += float(np.sum(residual**2))
        return total

    def _ode(self, _t: float, y: np.ndarray, k_vector: np.ndarray, z_structure: np.ndarray) -> np.ndarray:
        concentrations = np.maximum(y, 0.0)
        rates = np.zeros(self.n_reactions, dtype=float)
        for j, pattern in enumerate(self.rate_patterns):
            if z_structure[j] < 0.5:
                continue
            value = k_vector[j]
            for species_idx, power in pattern:
                value *= concentrations[species_idx] ** power
            rates[j] = value
        return self.S @ rates

    def _integrate_rk4(
        self,
        k_vector: np.ndarray,
        z_structure: np.ndarray,
        y0: np.ndarray,
        t_eval: np.ndarray,
    ) -> np.ndarray:
        y = np.zeros((self.n_species, len(t_eval)), dtype=float)
        y[:, 0] = np.maximum(y0, 0.0)
        for i in range(1, len(t_eval)):
            t0 = float(t_eval[i - 1])
            dt = float(t_eval[i] - t_eval[i - 1])
            state = y[:, i - 1]
            k1 = self._ode(t0, state, k_vector, z_structure)
            k2 = self._ode(t0 + dt / 2, state + dt * k1 / 2, k_vector, z_structure)
            k3 = self._ode(t0 + dt / 2, state + dt * k2 / 2, k_vector, z_structure)
            k4 = self._ode(t0 + dt, state + dt * k3, k_vector, z_structure)
            y[:, i] = np.maximum(state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6, 0.0)
        return y

    def _rate_pattern(self, reaction: Reaction) -> list[tuple[int, int]]:
        counts = Counter(self.s_map[s.name] for s in reaction.reactants)
        return list(counts.items())

    def _detect_reversible_pairs(self) -> dict[int, int]:
        reverse_map: dict[int, int] = {}
        signatures = [
            (
                tuple(sorted(s.name for s in reaction.reactants)),
                tuple(sorted(s.name for s in reaction.products)),
            )
            for reaction in self.reactions
        ]
        for i, (lhs_i, rhs_i) in enumerate(signatures):
            if i in reverse_map:
                continue
            for j in range(i + 1, len(signatures)):
                if j in reverse_map:
                    continue
                lhs_j, rhs_j = signatures[j]
                if lhs_i == rhs_j and rhs_i == lhs_j:
                    reverse_map[j] = i
                    break
        return reverse_map
