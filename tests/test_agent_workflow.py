from pathlib import Path
from tempfile import TemporaryDirectory

from pc_mcmc_cigp.agent_backend import ChemicalSpeciesSpec, ProjectStage, ReactionProjectSpec
from pc_mcmc_cigp.agent_backend.workflow import ReactionAgentWorkflow, load_runtime_config


def test_workflow_creates_plan_versions_and_waits_for_user_data():
    with TemporaryDirectory() as tmp:
        project=ReactionProjectSpec("p","A to P","combined","request",[ChemicalSpeciesSpec("A")],[ChemicalSpeciesSpec("P")])
        workflow=ReactionAgentWorkflow(Path(tmp)/"projects"); workflow.create_project(project)
        requests=workflow.prepare_experiment_plan(project)
        assert requests and project.stage == ProjectStage.WAITING_FOR_DATA
        root=Path(tmp)/"projects"/"p"/"experiment_requests"
        assert (root/"v001.json").exists() and (root/"v001.csv").exists()


def test_missing_runtime_config_keeps_api_disabled():
    config=load_runtime_config("path-that-does-not-exist.json")
    assert config["api_enabled"] is False and config["openai_model"] is None
