import csv
from pathlib import Path
from tempfile import TemporaryDirectory

from pc_mcmc_cigp.agent_backend import ChemicalSpeciesSpec, ExperimentalVariable, Observable, ReactionProjectSpec
from pc_mcmc_cigp.agent_backend.experiments import ExperimentDataValidator, ExperimentPlanner, write_experiment_template


def _project():
    return ReactionProjectSpec(
        project_id="rxn", title="A+B to P", objective="combined", raw_user_request="study A+B",
        reactants=[ChemicalSpeciesSpec("A", {"C": 1}), ChemicalSpeciesSpec("B", {"O": 1})],
        known_products=[ChemicalSpeciesSpec("P", {"C": 1, "O": 1})],
        controllable_variables=[ExperimentalVariable("temperature", "K", 300, 360)],
        observed_variables=[Observable("A"), Observable("B"), Observable("P")],
    )


def test_planner_writes_multitemperature_fillable_csv():
    with TemporaryDirectory() as tmp:
        project = _project(); requests = ExperimentPlanner().plan(project)
        path = write_experiment_template(project, requests, Path(tmp) / "template.csv")
        rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
        assert len(requests) == 5 and rows
        assert {row["temperature_K"] for row in rows} == {"300.0", "330.0", "360.0"}
        assert "P_mol_L" in rows[0]


def test_validator_accepts_complete_data_and_flags_weak_design():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.csv"
        path.write_text("experiment_id,time_s,temperature_K,A_mol_L,B_mol_L,P_mol_L\nE1,0,330,1,1,0\nE1,10,330,0.8,0.8,0.2\n", encoding="utf-8")
        report, rows = ExperimentDataValidator().validate(_project(), path)
        assert report.valid and len(rows) == 2
        assert report.identifiability_risks


def test_validator_rejects_negative_or_missing_concentration_data():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.csv"
        path.write_text("experiment_id,time_s,A_mol_L,B_mol_L\nE1,0,1,-1\n", encoding="utf-8")
        report, _ = ExperimentDataValidator().validate(_project(), path)
        assert not report.valid
        assert any("missing concentration" in item for item in report.errors)
