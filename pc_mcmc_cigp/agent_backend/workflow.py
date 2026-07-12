from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

from pc_mcmc_cigp.agent_backend.experiments import ExperimentDataValidator, ExperimentPlanner, write_experiment_template
from pc_mcmc_cigp.agent_backend.models import MechanismSpec, ProjectStage, ReactionProjectSpec
from pc_mcmc_cigp.agent_backend.services import AlgorithmService
from pc_mcmc_cigp.agent_backend.store import ProjectStore


class ReactionAgentWorkflow:
    """Deterministic workflow that a future LLM agent will call through tools."""

    def __init__(self, project_root: str | Path) -> None:
        self.store = ProjectStore(project_root)
        self.planner = ExperimentPlanner()
        self.validator = ExperimentDataValidator()
        self.algorithms = AlgorithmService()

    def create_project(self, project: ReactionProjectSpec) -> Path:
        return self.store.create(project)

    def prepare_experiment_plan(self, project: ReactionProjectSpec):
        if project.stage == ProjectStage.INTAKE and project.missing_information:
            self.store.transition(project, ProjectStage.NEEDS_CLARIFICATION, "missing information must be resolved")
            return []
        requests = self.planner.plan(project)
        if project.stage in {ProjectStage.INTAKE, ProjectStage.NEEDS_CLARIFICATION, ProjectStage.NEEDS_MORE_DATA}:
            self.store.transition(project, ProjectStage.EXPERIMENT_PLAN_READY)
        payload = [asdict(item) for item in requests]
        self.store.save_artifact(project.project_id, "experiment_requests", payload)
        template = self.store.next_version_path(project.project_id, "experiment_requests", "csv")
        write_experiment_template(project, requests, template)
        self.store.append_event(project.project_id, "experiment_template_saved", {"path": template.name})
        self.store.transition(project, ProjectStage.WAITING_FOR_DATA)
        return requests

    def ingest_dataset(self, project: ReactionProjectSpec, dataset_path: str | Path):
        if project.stage != ProjectStage.WAITING_FOR_DATA:
            raise ValueError("project must be waiting for data before ingestion")
        self.store.transition(project, ProjectStage.DATA_VALIDATION)
        report, normalized = self.validator.validate(project, dataset_path)
        self.store.save_artifact(project.project_id, "reports", asdict(report))
        if not report.valid:
            self.store.transition(project, ProjectStage.WAITING_FOR_DATA, "dataset validation failed")
            return report, []
        dataset_version = self.store.save_artifact(project.project_id, "datasets", normalized)
        report = replace(report, dataset_version=dataset_version.name)
        self.store.transition(project, ProjectStage.MECHANISM_PROPOSAL)
        return report, normalized

    def register_mechanism(self, project: ReactionProjectSpec, mechanism: MechanismSpec):
        if project.stage != ProjectStage.MECHANISM_PROPOSAL:
            raise ValueError("project is not ready for a mechanism proposal")
        compiled = self.algorithms.compile_mechanism(mechanism)
        self.store.save_artifact(project.project_id, "mechanisms", asdict(mechanism))
        self.store.save_artifact(project.project_id, "reports", asdict(compiled.report))
        if compiled.report.valid:
            self.store.transition(project, ProjectStage.WAITING_FOR_MECHANISM_APPROVAL)
        return compiled.report

    def approve_mechanism(self, project: ReactionProjectSpec, mechanism: MechanismSpec) -> MechanismSpec:
        if project.stage != ProjectStage.WAITING_FOR_MECHANISM_APPROVAL:
            raise ValueError("project is not waiting for mechanism approval")
        approved = replace(mechanism, approved=True)
        self.store.save_artifact(project.project_id, "mechanisms", asdict(approved))
        self.store.append_event(project.project_id, "mechanism_approved", {"mechanism_id": approved.mechanism_id})
        return approved


def load_runtime_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {"api_enabled": False, "openai_model": None, "openai_api_key_env": "OPENAI_API_KEY"}
    return json.loads(path.read_text(encoding="utf-8"))
