from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from pc_mcmc_cigp.agent_backend.http_api import ReactionAPI
from pc_mcmc_cigp.agent_backend.codecs import project_from_dict
from pc_mcmc_cigp.agent_backend.models import ProjectStage
from pc_mcmc_cigp.kinetics import create_kinetic_template


def _project():
    return {"project_id":"web-demo","title":"A to P","objective":"combined","raw_user_request":"discover and optimize","reactants":[{"name":"A","formula":{"C":1},"role":"reactant"}],"known_products":[{"name":"P","formula":{"C":1},"role":"product"}],"observed_variables":[{"species":"A"},{"species":"P"}]}


def test_http_contract_creates_project_plans_experiments_and_serves_dashboard():
    with TemporaryDirectory() as tmp:
        static=Path(tmp)/"static"; static.mkdir(); (static/"index.html").write_text("ok",encoding="utf-8")
        api=ReactionAPI(Path(tmp)/"projects",static)
        assert api.dispatch("GET","/api/health")[0]==200
        status, llm, _ = api.dispatch("GET", "/api/llm/status")
        assert status == 200 and llm["configured"] is False
        status, body, content_type = api.dispatch("GET", "/")
        assert status == 200 and body == b"ok" and content_type == "text/html"
        status,created,_=api.dispatch("POST","/api/projects",_project()); assert status==201
        status,plan,_=api.dispatch("POST","/api/projects/web-demo/experiment-plan"); assert status==200 and plan["requests"]
        status,dashboard,_=api.dispatch("GET","/api/projects/web-demo/dashboard"); assert status==200 and dashboard["stage"]=="waiting_for_data"


def test_http_contract_validates_uploaded_csv_and_compiles_mechanism():
    with TemporaryDirectory() as tmp:
        static=Path(tmp)/"static"; static.mkdir(); (static/"index.html").write_text("ok",encoding="utf-8")
        api=ReactionAPI(Path(tmp)/"projects",static); api.dispatch("POST","/api/projects",_project()); api.dispatch("POST","/api/projects/web-demo/experiment-plan")
        csv_text="experiment_id,time_s,temperature_K,A_mol_L,P_mol_L\nE1,0,300,1,0\nE1,10,300,.8,.2\n"
        status,result,_=api.dispatch("POST","/api/projects/web-demo/datasets/validate",{"csv_text":csv_text}); assert status==200 and result["report"]["valid"]
        mechanism={"mechanism_id":"m1","project_id":"web-demo","species":[{"name":"A","formula":{"C":1}},{"name":"P","formula":{"C":1}}],"steps":[{"step_id":"s1","reactants":{"A":1},"products":{"P":1}}]}
        status,result,_=api.dispatch("POST","/api/mechanisms/compile",mechanism); assert status==200 and result["report"]["valid"]
        status,result,_=api.dispatch("POST","/api/projects/web-demo/mechanisms/register",mechanism); assert status==200
        status,result,_=api.dispatch("POST","/api/projects/web-demo/mechanisms/approve",mechanism); assert status==200 and result["mechanism"]["approved"]


def test_http_cigp_route_respects_project_stage_and_returns_recommendation():
    with TemporaryDirectory() as tmp:
        static=Path(tmp)/"static"; static.mkdir(); (static/"index.html").write_text("ok",encoding="utf-8")
        api=ReactionAPI(Path(tmp)/"projects",static); api.dispatch("POST","/api/projects",_project())
        project=project_from_dict(api.workflow.store.load("web-demo"))
        for stage in [ProjectStage.EXPERIMENT_PLAN_READY,ProjectStage.WAITING_FOR_DATA,ProjectStage.DATA_VALIDATION,ProjectStage.MECHANISM_PROPOSAL,ProjectStage.WAITING_FOR_MECHANISM_APPROVAL,ProjectStage.MCMC_RUNNING,ProjectStage.MCMC_REVIEW]:
            api.workflow.store.transition(project,stage)
        X=[[1.,1.,320.,.01],[1.,1.,340.,.02],[1.,1.,360.,.04]]; physics=create_kinetic_template("simple_arrhenius"); y=physics.compute_mean(np.asarray(X),physics.W).ravel().tolist()
        mcmc={"run_id":"m","converged":True,"reaction_pip":{"s":.9},"pathway_pip":{},"top_reactions":["s"],"parameter_intervals":{},"posterior_predictive_metrics":{"rhat_max":1.01},"identifiability_warnings":[],"recommended_next_action":"fit_cigp"}
        payload={"mcmc_summary":mcmc,"template_name":"simple_arrhenius","X":X,"y":y,"bounds":{"A0":[.8,1.2],"B0":[.8,1.2],"temperature":[310,370],"time":[.005,.05]},"config":{"n_candidates":16}}
        status,result,_=api.dispatch("POST","/api/projects/web-demo/cigp/run",payload)
        assert status==200 and result["report"]["recommendation"]
