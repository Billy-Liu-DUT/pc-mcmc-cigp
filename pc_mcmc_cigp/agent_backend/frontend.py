from __future__ import annotations

from pathlib import Path

from pc_mcmc_cigp.agent_backend.models import ALLOWED_TRANSITIONS, AgentRuntimeConfig, FrontendDashboard, MechanismSpec
from pc_mcmc_cigp.agent_backend.store import ProjectStore


class FrontendReadModel:
    """Framework-neutral JSON read model for a future React/Vue/Streamlit UI."""

    def __init__(self, store: ProjectStore, runtime: AgentRuntimeConfig | None = None) -> None:
        self.store = store; self.runtime = runtime or AgentRuntimeConfig()

    def dashboard(self, project_id: str) -> FrontendDashboard:
        payload = self.store.load(project_id); stage = payload["stage"]
        project_dir = self.store.root / project_id
        counts = {folder: len(list((project_dir / folder).glob("v*.*"))) for folder in self.store.FOLDERS}
        from pc_mcmc_cigp.agent_backend.models import ProjectStage
        actions = tuple(sorted(item.value for item in ALLOWED_TRANSITIONS[ProjectStage(stage)]))
        return FrontendDashboard(project_id, stage, actions, counts, self.runtime.api_enabled)

    @staticmethod
    def mechanism_graph(spec: MechanismSpec) -> dict:
        nodes = [{"id": item.name, "role": item.role, "status": item.status} for item in spec.species]
        edges = []
        for step in spec.steps:
            for reactant in step.reactants:
                for product in step.products:
                    edges.append({"id": step.step_id, "source": reactant, "target": product, "rate_law": step.rate_law_family, "status": step.status})
        return {"nodes": nodes, "edges": edges, "approved": spec.approved}


def api_placeholder_status(runtime: AgentRuntimeConfig | None = None) -> dict:
    runtime = runtime or AgentRuntimeConfig()
    return {
        "configured": runtime.api_enabled and bool(runtime.openai_model),
        "model": runtime.openai_model,
        "credential_env": runtime.openai_api_key_env,
        "message": "OpenAI API is intentionally not configured" if not runtime.api_enabled else "OpenAI API configuration enabled",
    }
