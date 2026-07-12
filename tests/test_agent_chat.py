from pathlib import Path
from tempfile import TemporaryDirectory

from pc_mcmc_cigp.agent_backend.chat import ChatSessionStore
from pc_mcmc_cigp.agent_backend.llm_client import CompatibleResponsesClient, LLMProviderConfig


def test_chat_session_store_persists_messages_server_side():
    with TemporaryDirectory() as tmp:
        store=ChatSessionStore(Path(tmp)); session=store.create()
        store.append(session,"user","研究A到P")
        store.append(session,"assistant","请提供温度范围",{"intent":"clarify"})
        records=store.read(session)
        assert [row["role"] for row in records]==["user","assistant"]
        assert records[1]["metadata"]["intent"]=="clarify"


def test_llm_chat_returns_controlled_workflow_response():
    def transport(_url,_key,_payload):
        return {"output_text":'{"reply":"请提供温度范围。","intent":"clarify","missing_information":["temperature"],"suggested_actions":[],"project_draft":null}'}
    client=CompatibleResponsesClient(LLMProviderConfig("https://example.test/v1","gpt-5.5","secret"),transport)
    result=client.chat([{"role":"user","content":"研究A到P"}])
    assert result["intent"]=="clarify" and result["project_draft"] is None
