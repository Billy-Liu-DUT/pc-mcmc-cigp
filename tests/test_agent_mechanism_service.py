import numpy as np

from pc_mcmc_cigp.agent_backend import ChemicalSpeciesSpec, ElementaryStepSpec, MechanismSpec
from pc_mcmc_cigp.agent_backend.mechanism import MechanismCompiler, build_mcmc_dataset
from pc_mcmc_cigp.agent_backend.services import AlgorithmService


def _spec(approved=True):
    species = (ChemicalSpeciesSpec("A", {"C": 1}), ChemicalSpeciesSpec("I", {"C": 1}), ChemicalSpeciesSpec("P", {"C": 1}))
    steps = (
        ElementaryStepSpec("s1", {"A": 1}, {"I": 1}, rate_law_family="mass_action"),
        ElementaryStepSpec("s2", {"I": 1}, {"P": 1}, rate_law_family="mass_action"),
    )
    return MechanismSpec("m1", "p1", species, steps, approved=approved)


def _rows():
    return [
        {"experiment_id":"E1","replicate":1.,"time_s":0.,"A_mol_L":1.,"I_mol_L":0.,"P_mol_L":0.},
        {"experiment_id":"E1","replicate":1.,"time_s":0.1,"A_mol_L":0.9,"I_mol_L":0.09,"P_mol_L":0.01},
        {"experiment_id":"E1","replicate":1.,"time_s":0.2,"A_mol_L":0.8,"I_mol_L":0.16,"P_mol_L":0.04},
    ]


def test_compiler_builds_stoichiometric_matrix_and_path():
    compiler=MechanismCompiler(); compiled=compiler.compile(_spec())
    assert compiled.report.valid
    assert np.asarray(compiled.report.stoichiometric_matrix).shape == (3,2)
    assert compiler.candidate_pathways(compiled,["A"],["P"]) == [(0,1)]
    dataset=build_mcmc_dataset(compiled,_rows()); assert dataset[0]["data_matrix"].shape == (3,3)


def test_compiler_rejects_unbalanced_step():
    spec=MechanismSpec("bad","p",(ChemicalSpeciesSpec("A",{"C":1}),ChemicalSpeciesSpec("P",{"C":2})),(ElementaryStepSpec("x",{"A":1},{"P":1}),),True)
    assert not MechanismCompiler().compile(spec).report.valid


def test_service_requires_approval_and_returns_auditable_summary():
    service=AlgorithmService()
    try: service.run_pc_mcmc(_spec(False),_rows(),sources=["A"],targets=["P"],n_steps=20,burn_in=5,n_chains=2)
    except PermissionError: pass
    else: raise AssertionError("unapproved mechanism must not run")
    summary=service.run_pc_mcmc(_spec(True),_rows(),sources=["A"],targets=["P"],n_steps=30,burn_in=5,n_chains=2)
    assert set(summary.reaction_pip)=={"s1","s2"}
    assert "map_rmse" in summary.posterior_predictive_metrics
    assert summary.parameter_intervals
    assert "s1->s2" in summary.pathway_pip
