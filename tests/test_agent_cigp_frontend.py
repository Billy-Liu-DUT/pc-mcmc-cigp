from tempfile import TemporaryDirectory

import numpy as np

from pc_mcmc_cigp.agent_backend import (
    AgentRuntimeConfig, ChemicalSpeciesSpec, ElementaryStepSpec, MCMCSummary, MechanismSpec,
    ProjectStore, ReactionProjectSpec,
)
from pc_mcmc_cigp.agent_backend.cigp_service import CIGPService, available_template_contracts
from pc_mcmc_cigp.agent_backend.frontend import FrontendReadModel, api_placeholder_status


def _mcmc(converged=True):
    return MCMCSummary("m",converged,{"s":.9},{},("s",),{}, {"rhat_max":1.01},(),"fit_cigp")


def test_cigp_service_ranks_templates_and_recommends_conditions():
    X=np.array([[1.,1.,320.,.01],[1.,1.,340.,.02],[1.,1.,360.,.04]])
    from pc_mcmc_cigp.kinetics import create_kinetic_template
    physics=create_kinetic_template("simple_arrhenius"); y=physics.compute_mean(X,physics.W).ravel()
    service=CIGPService(); scores=service.rank_templates(X,y,["simple_arrhenius","robertson"])
    assert scores[0].compatible
    bounds={"A0":(.8,1.2),"B0":(.8,1.2),"temperature":(310.,370.),"time":(.005,.05)}
    report=service.fit_and_recommend(_mcmc(),"simple_arrhenius",X,y,bounds,n_candidates=16)
    assert report.recommendation and set(report.recommendation["conditions"])==set(bounds)


def test_cigp_requires_converged_mechanism_without_override():
    try: CIGPService().fit_and_recommend(_mcmc(False),"simple_arrhenius",np.ones((2,4)),np.ones(2),{"A0":(0,1),"B0":(0,1),"temperature":(300,400),"time":(0,1)})
    except PermissionError: pass
    else: raise AssertionError("unconverged mechanism must be gated")


def test_frontend_read_model_and_api_placeholder_are_framework_neutral():
    with TemporaryDirectory() as tmp:
        store=ProjectStore(tmp); project=ReactionProjectSpec("p","title","combined","request",[ChemicalSpeciesSpec("A")],[ChemicalSpeciesSpec("P")]); store.create(project)
        dashboard=FrontendReadModel(store).dashboard("p")
        assert dashboard.pages and not dashboard.api_enabled
        spec=MechanismSpec("m","p",(ChemicalSpeciesSpec("A"),ChemicalSpeciesSpec("P")),(ElementaryStepSpec("s",{"A":1},{"P":1}),),False)
        graph=FrontendReadModel.mechanism_graph(spec); assert len(graph["nodes"])==2 and graph["edges"]
        assert not api_placeholder_status(AgentRuntimeConfig())["configured"]
        assert available_template_contracts()
