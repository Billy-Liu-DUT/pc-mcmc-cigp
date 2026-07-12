from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np
try:
    from scipy.integrate import solve_ivp
except ImportError:  # pragma: no cover - exercised in minimal environments
    solve_ivp = None

from pc_mcmc_cigp.reactions import Reaction, Species
from pc_mcmc_cigp.kinetics import MassActionRate, RateLaw


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
        self.rate_laws: list[RateLaw] = []
        self.parameter_slices: list[slice] = []
        offset = 0
        for reaction, pattern in zip(self.reactions, self.rate_patterns):
            law = reaction.rate_law or MassActionRate(dict(pattern))
            self.rate_laws.append(law)
            width = len(law.parameter_names)
            self.parameter_slices.append(slice(offset, offset + width))
            offset += width
        self.n_rate_parameters = offset
        self.rate_parameter_names = tuple(
            f"r{i}_{name}" for i, law in enumerate(self.rate_laws) for name in law.parameter_names
        )
        self.rate_parameter_bounds = np.asarray(
            [bound for law in self.rate_laws for bound in law.parameter_bounds], dtype=float
        ) if self.rate_laws else np.empty((0, 2))

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
        temperature: float | None = None,
        context: dict[str, float] | None = None,
    ) -> np.ndarray:
        t_eval = np.asarray(t_eval, dtype=float)
        y0 = np.asarray(y0, dtype=float)
        k_arr = np.asarray(k_vector, dtype=float)
        z_arr = np.asarray(z_structure, dtype=float)
        if len(k_arr) not in {self.n_reactions, self.n_rate_parameters}:
            raise ValueError(f"expected {self.n_rate_parameters} rate parameters, got {len(k_arr)}")
        simulation_context = dict(context or {})
        if temperature is not None:
            simulation_context["temperature"] = float(temperature)
        def needs_temperature(law):
            if type(law).__name__ == "ArrheniusRate":
                return True
            return any(needs_temperature(getattr(law, name)) for name in ("forward", "reverse") if hasattr(law, name))
        if any(needs_temperature(law) for law in self.rate_laws) and "temperature" not in simulation_context:
            raise ValueError("Arrhenius rate laws require a temperature")
        if self.solver_backend in {"auto", "scipy"} and solve_ivp is not None:
            try:
                solution = solve_ivp(
                    lambda t, y: self._ode(t, y, k_arr, z_arr, simulation_context),
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
        return self._integrate_rk4(k_arr, z_arr, y0, t_eval, simulation_context)

    def calculate_loss(self, k_vector: Sequence[float], z_structure: Sequence[float], dataset: Sequence[dict]) -> float:
        total = 0.0
        for experiment in dataset:
            pred = self.simulate(
                k_vector, z_structure, experiment["y0_full"], experiment["t"],
                temperature=experiment.get("temperature"), context=experiment.get("conditions"),
            )
            obs_indices = experiment.get("obs_indices", list(range(self.n_species)))
            residual = pred[np.asarray(obs_indices), :] - np.asarray(experiment["data_matrix"], dtype=float)
            total += float(np.sum(residual**2))
        return total

    def _ode(self, _t: float, y: np.ndarray, k_vector: np.ndarray, z_structure: np.ndarray, context=None) -> np.ndarray:
        concentrations = np.maximum(y, 0.0)
        rates = np.zeros(self.n_reactions, dtype=float)
        legacy_layout = len(k_vector) == self.n_reactions and self.n_rate_parameters != self.n_reactions
        for j, law in enumerate(self.rate_laws):
            if z_structure[j] < 0.5:
                continue
            params = np.asarray([k_vector[j]]) if legacy_layout else k_vector[self.parameter_slices[j]]
            rates[j] = law.rate(concentrations, params, context)
        return self.S @ rates

    def _integrate_rk4(
        self,
        k_vector: np.ndarray,
        z_structure: np.ndarray,
        y0: np.ndarray,
        t_eval: np.ndarray,
        context: dict[str, float] | None = None,
    ) -> np.ndarray:
        y = np.zeros((self.n_species, len(t_eval)), dtype=float)
        y[:, 0] = np.maximum(y0, 0.0)
        for i in range(1, len(t_eval)):
            t0 = float(t_eval[i - 1])
            dt = float(t_eval[i] - t_eval[i - 1])
            state = y[:, i - 1]
            k1 = self._ode(t0, state, k_vector, z_structure, context)
            k2 = self._ode(t0 + dt / 2, state + dt * k1 / 2, k_vector, z_structure, context)
            k3 = self._ode(t0 + dt / 2, state + dt * k2 / 2, k_vector, z_structure, context)
            k4 = self._ode(t0 + dt, state + dt * k3, k_vector, z_structure, context)
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
