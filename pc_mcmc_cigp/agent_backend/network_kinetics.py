from __future__ import annotations

import numpy as np

from pc_mcmc_cigp.agent_backend.mechanism import CompiledMechanism, MechanismCompiler
from pc_mcmc_cigp.agent_backend.models import MCMCSummary, MechanismSpec
from pc_mcmc_cigp.kinetics import KineticModel, KineticParameter, SimulationResult


class CompiledNetworkKinetics(KineticModel):
    """CIGP physics mean generated directly from an approved PC-MCMC network."""

    def __init__(
        self, compiled: CompiledMechanism, target_species: str, inclusion_mask=None, defaults=None
    ) -> None:
        if not compiled.report.valid:
            raise ValueError("cannot create kinetics from an invalid mechanism")
        self.compiled = compiled
        self.species_names = compiled.report.species_order
        if target_species not in self.species_names:
            raise ValueError(f"unknown target species {target_species!r}")
        self.target_index = self.species_names.index(target_species)
        self.requires_temperature = any("log10_A" in name for name in compiled.engine.rate_parameter_names)
        self.input_names = (
            tuple(f"{name}0" for name in self.species_names)
            + (("temperature",) if self.requires_temperature else ())
            + ("time",)
        )
        bounds = compiled.engine.rate_parameter_bounds
        defaults = np.asarray(
            defaults if defaults is not None else [self._default(bound) for bound in bounds], dtype=float
        )
        parameters = [
            KineticParameter(name, float(default), tuple(map(float, bound)))
            for name, default, bound in zip(compiled.engine.rate_parameter_names, defaults, bounds)
        ]
        super().__init__(parameters)
        mask = (
            np.ones(compiled.engine.n_reactions)
            if inclusion_mask is None
            else np.asarray(inclusion_mask, dtype=float)
        )
        if mask.shape != (compiled.engine.n_reactions,):
            raise ValueError("inclusion mask must match reaction count")
        self.inclusion_mask = mask
        self.template_source = "mcmc_compiled"

    @staticmethod
    def _default(bound):
        lo, hi = map(float, bound)
        if lo > 0 and hi / lo > 100:
            return np.sqrt(lo * hi)
        return (lo + hi) / 2

    def simulate(self, X, W=None) -> SimulationResult:
        X = self.validate_X(X)
        W = self.W if W is None else np.asarray(W, dtype=float)
        self.validate_parameters(W)
        values, trajectories, solvers = [], [], []
        n_species = len(self.species_names)
        for row in X:
            y0 = row[:n_species]
            time = float(row[-1])
            temperature = float(row[-2]) if self.requires_temperature else None
            if time == 0:
                trajectory = np.column_stack([y0, y0])
                solver = "initial"
            else:
                trajectory = self.compiled.engine.simulate(
                    W, self.inclusion_mask, y0, [0.0, time], temperature=temperature
                )
                solver = self.compiled.engine.last_solver_backend
            trajectories.append(trajectory)
            values.append(trajectory[self.target_index, -1])
            solvers.append(solver)
        return SimulationResult(
            np.asarray(values).reshape(-1, 1), tuple(trajectories), True, "+".join(sorted(set(solvers)))
        )


def compile_posterior_kinetics(
    mechanism: MechanismSpec,
    summary: MCMCSummary,
    target_species: str,
    *,
    inclusion_threshold: float = 0.5,
) -> CompiledNetworkKinetics:
    compiled = MechanismCompiler().compile(mechanism)
    mask = [summary.reaction_pip.get(step_id, 0.0) >= inclusion_threshold for step_id in compiled.step_ids]
    if not any(mask):
        raise ValueError("no reaction passes the posterior inclusion threshold")
    defaults = []
    for name, bound in zip(compiled.engine.rate_parameter_names, compiled.engine.rate_parameter_bounds):
        interval = summary.parameter_intervals.get(name)
        defaults.append(
            (interval[0] + interval[1]) / 2 if interval else CompiledNetworkKinetics._default(bound)
        )
    return CompiledNetworkKinetics(compiled, target_species, mask, defaults)
