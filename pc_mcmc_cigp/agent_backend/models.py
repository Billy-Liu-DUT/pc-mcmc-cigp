from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal


class ProjectStage(str, Enum):
    INTAKE = "intake"
    NEEDS_CLARIFICATION = "needs_clarification"
    EXPERIMENT_PLAN_READY = "experiment_plan_ready"
    WAITING_FOR_DATA = "waiting_for_data"
    DATA_MAPPING = "data_mapping"
    DATA_VALIDATION = "data_validation"
    MECHANISM_PROPOSAL = "mechanism_proposal"
    WAITING_FOR_MECHANISM_APPROVAL = "waiting_for_mechanism_approval"
    MCMC_RUNNING = "mcmc_running"
    MCMC_REVIEW = "mcmc_review"
    NEEDS_MORE_DATA = "needs_more_data"
    CIGP_RUNNING = "cigp_running"
    CIGP_MODEL_COMPILATION = "cigp_model_compilation"
    CIGP_FITTING = "cigp_fitting"
    OPTIMIZATION_READY = "optimization_ready"
    WAITING_FOR_NEXT_EXPERIMENT = "waiting_for_next_experiment"
    FINAL_REVIEW = "final_review"


class WorkflowMode(str, Enum):
    MECHANISM_ONLY = "mechanism_only"
    OPTIMIZATION_ONLY = "optimization_only"
    COUPLED = "coupled"
    ITERATIVE_COUPLED = "iterative_coupled"


ALLOWED_TRANSITIONS: dict[ProjectStage, set[ProjectStage]] = {
    ProjectStage.INTAKE: {ProjectStage.NEEDS_CLARIFICATION, ProjectStage.EXPERIMENT_PLAN_READY},
    ProjectStage.NEEDS_CLARIFICATION: {ProjectStage.EXPERIMENT_PLAN_READY},
    ProjectStage.EXPERIMENT_PLAN_READY: {ProjectStage.WAITING_FOR_DATA},
    ProjectStage.WAITING_FOR_DATA: {ProjectStage.DATA_MAPPING, ProjectStage.DATA_VALIDATION},
    ProjectStage.DATA_MAPPING: {ProjectStage.WAITING_FOR_DATA, ProjectStage.DATA_VALIDATION},
    ProjectStage.DATA_VALIDATION: {ProjectStage.WAITING_FOR_DATA, ProjectStage.MECHANISM_PROPOSAL, ProjectStage.CIGP_MODEL_COMPILATION},
    ProjectStage.MECHANISM_PROPOSAL: {ProjectStage.WAITING_FOR_MECHANISM_APPROVAL},
    ProjectStage.WAITING_FOR_MECHANISM_APPROVAL: {ProjectStage.MECHANISM_PROPOSAL, ProjectStage.MCMC_RUNNING},
    ProjectStage.MCMC_RUNNING: {ProjectStage.MCMC_REVIEW},
    ProjectStage.MCMC_REVIEW: {ProjectStage.NEEDS_MORE_DATA, ProjectStage.CIGP_MODEL_COMPILATION, ProjectStage.CIGP_RUNNING, ProjectStage.FINAL_REVIEW},
    ProjectStage.NEEDS_MORE_DATA: {ProjectStage.EXPERIMENT_PLAN_READY, ProjectStage.WAITING_FOR_DATA},
    ProjectStage.CIGP_MODEL_COMPILATION: {ProjectStage.CIGP_FITTING, ProjectStage.MCMC_REVIEW},
    ProjectStage.CIGP_FITTING: {ProjectStage.OPTIMIZATION_READY, ProjectStage.MCMC_REVIEW, ProjectStage.DATA_VALIDATION},
    ProjectStage.CIGP_RUNNING: {ProjectStage.OPTIMIZATION_READY},
    ProjectStage.OPTIMIZATION_READY: {ProjectStage.WAITING_FOR_NEXT_EXPERIMENT, ProjectStage.CIGP_FITTING, ProjectStage.FINAL_REVIEW},
    ProjectStage.WAITING_FOR_NEXT_EXPERIMENT: {ProjectStage.DATA_MAPPING, ProjectStage.FINAL_REVIEW},
    ProjectStage.FINAL_REVIEW: {ProjectStage.EXPERIMENT_PLAN_READY, ProjectStage.MECHANISM_PROPOSAL, ProjectStage.CIGP_FITTING},
}


@dataclass(frozen=True)
class ChemicalSpeciesSpec:
    name: str
    formula: dict[str, int] = field(default_factory=dict)
    charge: int = 0
    role: str = "unknown"
    status: Literal["known", "user_proposed", "llm_candidate"] = "known"


@dataclass(frozen=True)
class ExperimentalVariable:
    name: str
    unit: str
    lower: float | None = None
    upper: float | None = None


@dataclass(frozen=True)
class Observable:
    species: str
    unit: str = "mol/L"
    required: bool = True
    detection_limit: float | None = None


@dataclass
class ReactionProjectSpec:
    project_id: str
    title: str
    objective: str
    raw_user_request: str
    reactants: list[ChemicalSpeciesSpec]
    known_products: list[ChemicalSpeciesSpec]
    suspected_intermediates: list[ChemicalSpeciesSpec] = field(default_factory=list)
    catalysts: list[ChemicalSpeciesSpec] = field(default_factory=list)
    solvents: list[ChemicalSpeciesSpec] = field(default_factory=list)
    controllable_variables: list[ExperimentalVariable] = field(default_factory=list)
    observed_variables: list[Observable] = field(default_factory=list)
    known_conditions: dict[str, Any] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    candidate_reaction_families: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    stage: ProjectStage = ProjectStage.INTAKE
    workflow_mode: WorkflowMode = WorkflowMode.COUPLED
    schema_version: str = "1.0"

    def transition(self, target: ProjectStage) -> None:
        if target not in ALLOWED_TRANSITIONS[self.stage]:
            raise ValueError(f"invalid project transition: {self.stage.value} -> {target.value}")
        self.stage = target

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["stage"] = self.stage.value
        payload["workflow_mode"] = self.workflow_mode.value
        return payload


@dataclass(frozen=True)
class ColumnMapping:
    source: str
    target: str
    conversion: str = "identity"
    confidence: float = 1.0
    requires_confirmation: bool = False


@dataclass(frozen=True)
class DataMappingReport:
    valid: bool
    mappings: tuple[ColumnMapping, ...]
    unresolved_columns: tuple[str, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    normalized_rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ExperimentRequest:
    experiment_id: str
    purpose: str
    variables: dict[str, float]
    sampling_times: tuple[float, ...]
    required_observables: tuple[str, ...]
    optional_observables: tuple[str, ...] = ()
    replicates: int = 2
    rationale: str = ""


@dataclass(frozen=True)
class DataValidationReport:
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    normalized_dataset_path: str | None
    species_coverage: dict[str, float]
    mass_balance_residuals: dict[str, float]
    identifiability_risks: tuple[str, ...]
    dataset_version: str | None = None


@dataclass(frozen=True)
class ElementaryStepSpec:
    step_id: str
    reactants: dict[str, float]
    products: dict[str, float]
    catalysts: dict[str, float] = field(default_factory=dict)
    rate_law_family: str = "mass_action"
    reversible: bool = False
    prior_probability: float = 0.1
    evidence: str = ""
    status: Literal["known", "user_proposed", "llm_candidate"] = "llm_candidate"


@dataclass(frozen=True)
class MechanismSpec:
    mechanism_id: str
    project_id: str
    species: tuple[ChemicalSpeciesSpec, ...]
    steps: tuple[ElementaryStepSpec, ...]
    approved: bool = False
    schema_version: str = "1.0"


@dataclass(frozen=True)
class CompilationReport:
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    species_order: tuple[str, ...]
    stoichiometric_matrix: list[list[float]]
    parameter_names: tuple[str, ...]


@dataclass(frozen=True)
class MCMCSummary:
    run_id: str
    converged: bool
    reaction_pip: dict[str, float]
    pathway_pip: dict[str, float]
    top_reactions: tuple[str, ...]
    parameter_intervals: dict[str, tuple[float, float]]
    posterior_predictive_metrics: dict[str, float]
    identifiability_warnings: tuple[str, ...]
    recommended_next_action: str


@dataclass(frozen=True)
class CIGPReport:
    run_id: str
    template_name: str
    training_rows: int
    objective: str
    best_observed: float
    recommendation: dict[str, Any] | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class TemplateScore:
    template_name: str
    compatible: bool
    rmse: float | None
    reason: str


@dataclass(frozen=True)
class AgentRuntimeConfig:
    openai_model: str | None = None
    openai_api_key_env: str = "OPENAI_API_KEY"
    openai_base_url_env: str = "OPENAI_BASE_URL"
    llm_max_concurrency: int = 1
    api_enabled: bool = False


@dataclass(frozen=True)
class FrontendDashboard:
    project_id: str
    stage: str
    available_actions: tuple[str, ...]
    artifact_counts: dict[str, int]
    api_enabled: bool
    pages: tuple[str, ...] = (
        "project_wizard", "experiment_plan", "data_quality", "mechanism_graph", "mcmc_diagnostics", "cigp_optimization"
    )
