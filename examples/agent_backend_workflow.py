"""Run the deterministic part of the future reaction agent without an API key."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pc_mcmc_cigp.agent_backend import ChemicalSpeciesSpec, Observable, ReactionProjectSpec
from pc_mcmc_cigp.agent_backend.workflow import ReactionAgentWorkflow


def main() -> None:
    project = ReactionProjectSpec(
        project_id="styrene_epoxidation", title="Styrene epoxidation", objective="combined",
        raw_user_request="Discover the kinetic mechanism and optimize epoxide yield",
        reactants=[ChemicalSpeciesSpec("styrene", {"C": 8, "H": 8}, role="reactant")],
        known_products=[ChemicalSpeciesSpec("epoxide", {"C": 8, "H": 8, "O": 1}, role="product")],
        observed_variables=[Observable("styrene"), Observable("epoxide")],
    )
    with TemporaryDirectory() as tmp:
        workflow = ReactionAgentWorkflow(Path(tmp) / "projects")
        root = workflow.create_project(project)
        requests = workflow.prepare_experiment_plan(project)
        print(f"Project: {root}")
        print(f"Stage: {project.stage.value}")
        print(f"Experiment requests: {len(requests)}")
        print("OpenAI API was not used.")


if __name__ == "__main__":
    main()
