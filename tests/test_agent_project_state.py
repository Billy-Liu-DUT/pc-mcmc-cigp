from tempfile import TemporaryDirectory

from pc_mcmc_cigp.agent_backend import ChemicalSpeciesSpec, ProjectStage, ProjectStore, ReactionProjectSpec


def _project():
    return ReactionProjectSpec(
        project_id="demo", title="Demo", objective="combined", raw_user_request="optimize A to P",
        reactants=[ChemicalSpeciesSpec("A", {"C": 1}, role="reactant")],
        known_products=[ChemicalSpeciesSpec("P", {"C": 1}, role="product")],
    )


def test_project_store_versions_artifacts_and_records_transitions():
    with TemporaryDirectory() as tmp:
        store = ProjectStore(tmp); project = _project(); store.create(project)
        store.transition(project, ProjectStage.EXPERIMENT_PLAN_READY)
        first = store.save_artifact("demo", "reports", {"ok": True})
        second = store.save_artifact("demo", "reports", {"ok": False})
        assert first.name == "v001.json" and second.name == "v002.json"
        assert store.load("demo")["stage"] == "experiment_plan_ready"


def test_project_state_rejects_invalid_transition():
    project = _project()
    try:
        project.transition(ProjectStage.MCMC_RUNNING)
    except ValueError as exc:
        assert "invalid project transition" in str(exc)
    else:
        raise AssertionError("invalid transition must fail")
