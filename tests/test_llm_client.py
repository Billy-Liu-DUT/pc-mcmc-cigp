from pc_mcmc_cigp.agent_backend.llm_client import CompatibleResponsesClient, LLMProviderConfig


def _transport(_url,_key,payload):
    if "Structure reaction-kinetics" in payload.get("instructions",""):
        text='''{"project_id":"x","title":"A to P","objective":"combined","raw_user_request":"","reactants":[{"name":"A","formula":{"C":1},"role":"reactant"}],"known_products":[{"name":"P","formula":{"C":1},"role":"product"}],"suspected_intermediates":[],"catalysts":[],"solvents":[],"controllable_variables":[],"observed_variables":[{"species":"A"},{"species":"P"}],"known_conditions":{},"constraints":[],"candidate_reaction_families":[],"uncertainties":[],"missing_information":[]}'''
    else:
        text='''{"mechanism_id":"m1","project_id":"x","species":[{"name":"A","formula":{"C":1}},{"name":"P","formula":{"C":1}}],"steps":[{"step_id":"s1","reactants":{"A":1},"products":{"P":1},"rate_law_family":"mass_action","status":"llm_candidate"}],"approved":false}'''
    return {"output":[{"content":[{"type":"output_text","text":text}]}]}


def test_compatible_client_converts_json_to_strict_contracts():
    client=CompatibleResponsesClient(LLMProviderConfig("https://example.test/v1","gpt-5.5","secret"),_transport)
    project=client.parse_project("discover A to P","p1"); mechanism=client.propose_mechanism(project.to_dict())
    assert project.project_id=="p1" and not mechanism.approved and mechanism.steps[0].status=="llm_candidate"
