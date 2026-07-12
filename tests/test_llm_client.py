from pc_mcmc_cigp.agent_backend.llm_client import CompatibleResponsesClient, LLMProviderConfig
import threading
import time


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


def test_provider_calls_are_serialized_to_concurrency_one():
    active=0; maximum=0; guard=threading.Lock()
    def slow_transport(_url,_key,_payload):
        nonlocal active,maximum
        with guard: active+=1; maximum=max(maximum,active)
        time.sleep(0.03)
        with guard: active-=1
        return {"output_text":'{"project_id":"x","title":"A","objective":"combined","reactants":[],"known_products":[],"missing_information":[]}' }
    client=CompatibleResponsesClient(LLMProviderConfig("https://example.test/v1","gpt-5.5","secret"),slow_transport)
    threads=[threading.Thread(target=lambda: client.parse_project("A","p")) for _ in range(3)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    assert maximum==1
