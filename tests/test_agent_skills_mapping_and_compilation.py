import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from pc_mcmc_cigp.agent_backend.data_mapping import BenchmarkDataMapper, build_cigp_training_data
from pc_mcmc_cigp.agent_backend.models import (
    ChemicalSpeciesSpec,
    ElementaryStepSpec,
    MCMCSummary,
    MechanismSpec,
    ReactionProjectSpec,
)
from pc_mcmc_cigp.agent_backend.network_kinetics import compile_posterior_kinetics
from pc_mcmc_cigp.agent_backend.skills import SkillRuntime
from pc_mcmc_cigp.agent_backend.control import ExperimentLoopController, MCMCTuningPolicy
from pc_mcmc_cigp.agent_backend.models import WorkflowMode


def _project():
    return ReactionProjectSpec(
        "p",
        "A oxidation",
        "combined",
        "discover and optimize",
        [ChemicalSpeciesSpec("A", {"C": 1})],
        [ChemicalSpeciesSpec("P", {"C": 1})],
    )


def test_mapper_converts_benchmark_aliases_and_builds_cigp_matrix():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["experiment", "time_min", "temp_C", "A_mol_L", "P_mol_L"]
            )
            writer.writeheader()
            writer.writerow({"experiment": "E1", "time_min": 0, "temp_C": 25, "A_mol_L": 1, "P_mol_L": 0})
            writer.writerow(
                {"experiment": "E1", "time_min": 1, "temp_C": 25, "A_mol_L": 0.5, "P_mol_L": 0.5}
            )
        report = BenchmarkDataMapper().map_csv(_project(), path)
        assert report.valid and report.normalized_rows[1]["time_s"] == 60
        assert np.isclose(report.normalized_rows[0]["temperature_K"], 298.15)


def test_posterior_network_compiles_to_cigp_physics_and_training_contract():
    species = (ChemicalSpeciesSpec("A", {"C": 1}), ChemicalSpeciesSpec("P", {"C": 1}))
    mechanism = MechanismSpec("m", "p", species, (ElementaryStepSpec("s1", {"A": 1}, {"P": 1}),), True)
    summary = MCMCSummary("run", True, {"s1": 0.95}, {}, ("s1",), {"r0_k": (0.5, 1.5)}, {}, (), "fit_cigp")
    model = compile_posterior_kinetics(mechanism, summary, "P")
    values = model.compute_mean(np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]]), model.W)
    assert values.shape == (2, 1) and values[0, 0] == 0 and values[1, 0] > 0
    rows = [
        {"experiment_id": "E1", "time_s": 0.0, "A_mol_L": 1.0, "P_mol_L": 0.0},
        {"experiment_id": "E1", "time_s": 1.0, "A_mol_L": 0.4, "P_mol_L": 0.6},
    ]
    X, y = build_cigp_training_data(rows, model, "P_mol_L")
    assert X.shape == (2, 3) and np.allclose(y, [0, 0.6])


def test_prompt_eval_routes_and_all_schemas_are_readable():
    runtime = SkillRuntime()
    cases = json.loads(open("evals/agent_prompt_cases.json", encoding="utf-8").read())
    assert all(runtime.route(case["stage"], case["message"]).name == case["expected_skill"] for case in cases)
    assert all(runtime.schema(skill)["additionalProperties"] is False for skill in runtime.registry.list())


def test_agent_may_tune_numerics_but_scientific_space_requires_approval():
    review = MCMCTuningPolicy().review(
        {"n_steps": 2000, "candidate_intermediates": ["I"], "delete_bad_rows": True}
    )
    assert review.automatic == {"n_steps": 2000}
    assert "candidate_intermediates" in review.requires_user_approval
    assert "delete_bad_rows" in review.rejected
    summary = MCMCSummary("run", True, {}, {}, (), {}, {}, (), "fit_cigp")
    assert (
        ExperimentLoopController().after_mcmc(summary, WorkflowMode.COUPLED)
        == "compile_mcmc_network_for_cigp"
    )
