from __future__ import annotations

from uuid import uuid4

import numpy as np

from pc_mcmc_cigp.agent_backend.mechanism import MechanismCompiler, build_mcmc_dataset
from pc_mcmc_cigp.agent_backend.models import MCMCSummary, MechanismSpec
from pc_mcmc_cigp.discovery import MCMCConfig, run_multiple_chains


class AlgorithmService:
    """Stable tool-facing boundary around mechanism compilation and numerical inference."""

    def __init__(self) -> None:
        self.compiler = MechanismCompiler()

    def compile_mechanism(self, spec: MechanismSpec):
        return self.compiler.compile(spec)

    def run_pc_mcmc(
        self, spec: MechanismSpec, normalized_rows, *, sources, targets,
        n_steps: int = 1000, burn_in: int = 200, n_chains: int = 4, random_state: int = 0,
    ) -> MCMCSummary:
        if not spec.approved:
            raise PermissionError("mechanism must be explicitly approved before PC-MCMC")
        compiled = self.compiler.compile(spec)
        if not compiled.report.valid:
            raise ValueError("mechanism compilation failed: " + "; ".join(compiled.report.errors))
        dataset = build_mcmc_dataset(compiled, normalized_rows)
        pathways = self.compiler.candidate_pathways(compiled, sources, targets)
        config = MCMCConfig(n_steps=n_steps, burn_in=burn_in, random_state=random_state, enable_thermo_constraints=False)
        result = run_multiple_chains(compiled.engine, dataset, config, n_chains=n_chains, candidate_pathways=pathways)
        reaction_pip = {step_id: float(value) for step_id, value in zip(compiled.step_ids, result.posterior_inclusion_probabilities)}
        sampled_pathway_pip = result.chain_diagnostics.get("pathway_pip", np.empty(0)) if result.chain_diagnostics else np.empty(0)
        pathway_pip = {
            "->".join(compiled.step_ids[i] for i in path): float(probability)
            for path, probability in zip(pathways, sampled_pathway_pip)
        }
        intervals = {}
        if result.chain_diagnostics:
            q05 = result.chain_diagnostics.get("parameter_q05", [])
            q95 = result.chain_diagnostics.get("parameter_q95", [])
            intervals = {name: (float(lo), float(hi)) for name, lo, hi in zip(compiled.engine.rate_parameter_names, q05, q95)}
        rhat = float(result.diagnostics.get("rhat_max", np.inf)); ess = float(result.diagnostics.get("ess_min", 0.0))
        converged = bool(np.isfinite(rhat) and rhat <= 1.05 and ess >= 20)
        warnings = []
        if not converged: warnings.append(f"MCMC convergence is not yet adequate (R-hat={rhat:.3g}, ESS={ess:.3g})")
        if result.diagnostics.get("invalid_evaluations", 0): warnings.append("Some proposed parameter sets produced invalid ODE evaluations")
        return MCMCSummary(
            run_id=f"mcmc_{uuid4().hex[:12]}", converged=converged, reaction_pip=reaction_pip,
            pathway_pip=pathway_pip, top_reactions=tuple(name for name, value in reaction_pip.items() if value >= config.selection_threshold),
            parameter_intervals=intervals, posterior_predictive_metrics={"map_rmse": float(result.diagnostics.get("map_rmse", np.nan)), "rhat_max": rhat, "ess_min": ess},
            identifiability_warnings=tuple(warnings),
            recommended_next_action="fit_cigp" if converged else "collect_discriminating_data_or_run_longer_chains",
        )
