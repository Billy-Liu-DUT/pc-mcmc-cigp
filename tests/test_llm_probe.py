import os

from scripts import probe_llm_api


def test_probe_refuses_to_run_without_environment_credentials(monkeypatch=None):
    old_key, old_base = os.environ.pop("OPENAI_API_KEY", None), os.environ.pop("OPENAI_BASE_URL", None)
    try:
        assert probe_llm_api.main() == 2
    finally:
        if old_key is not None: os.environ["OPENAI_API_KEY"] = old_key
        if old_base is not None: os.environ["OPENAI_BASE_URL"] = old_base
