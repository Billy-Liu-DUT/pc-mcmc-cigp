from __future__ import annotations

from dataclasses import dataclass

from pc_mcmc_cigp.agent_backend.models import MCMCSummary, WorkflowMode


@dataclass(frozen=True)
class AdjustmentReview:
    automatic: dict
    requires_user_approval: dict
    rejected: dict


class MCMCTuningPolicy:
    """Separate numerical tuning from changes to the scientific hypothesis space."""

    NUMERICAL = {
        "n_steps",
        "burn_in",
        "n_chains",
        "random_state",
        "step_size",
        "sigma_likelihood",
        "solver",
        "rtol",
        "atol",
        "parallel_temperatures",
    }
    SCIENTIFIC = {
        "candidate_intermediates",
        "candidate_steps",
        "prior_probabilities",
        "reversibility",
        "rate_law_family",
        "parameter_bounds",
        "sources",
        "targets",
    }

    def review(self, changes: dict) -> AdjustmentReview:
        automatic = {key: value for key, value in changes.items() if key in self.NUMERICAL}
        approval = {key: value for key, value in changes.items() if key in self.SCIENTIFIC}
        rejected = {
            key: value for key, value in changes.items() if key not in self.NUMERICAL | self.SCIENTIFIC
        }
        return AdjustmentReview(automatic, approval, rejected)


class ExperimentLoopController:
    """Choose the next loop without asking the LLM to invent workflow state."""

    def after_mcmc(self, summary: MCMCSummary, mode: WorkflowMode) -> str:
        if not summary.converged:
            return "diagnose_or_collect_discriminating_data"
        if mode == WorkflowMode.MECHANISM_ONLY:
            return "final_review"
        return "compile_mcmc_network_for_cigp"

    def after_new_data(self, *, systematic_model_mismatch: bool, within_design_bounds: bool) -> str:
        if systematic_model_mismatch:
            return "return_to_mechanism_proposal"
        if not within_design_bounds:
            return "validate_extrapolation_then_refit_cigp"
        return "update_cigp"

    def next_experiment_purpose(self, *, mechanism_uncertainty: float, objective_uncertainty: float) -> str:
        if mechanism_uncertainty >= objective_uncertainty:
            return "mechanism_discrimination"
        return "optimization"
