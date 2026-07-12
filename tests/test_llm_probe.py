import os
from urllib.error import HTTPError
from io import BytesIO

from scripts import probe_llm_api


def test_probe_refuses_to_run_without_environment_credentials(monkeypatch=None):
    old_key, old_base = os.environ.pop("OPENAI_API_KEY", None), os.environ.pop("OPENAI_BASE_URL", None)
    try:
        assert probe_llm_api.main() == 2
    finally:
        if old_key is not None: os.environ["OPENAI_API_KEY"] = old_key
        if old_base is not None: os.environ["OPENAI_BASE_URL"] = old_base


def test_probe_preserves_non_json_403_diagnostics(monkeypatch=None):
    class Headers(dict):
        def get(self, key, default=None): return super().get(key, default)
    error=HTTPError("https://example.test/v1/models",403,"Forbidden",Headers({"Server":"cloudflare","Content-Type":"text/html","cf-ray":"abc"}),BytesIO(b"<html>Access denied</html>"))
    original=probe_llm_api.urlopen
    try:
        probe_llm_api.urlopen=lambda *_args,**_kwargs: (_ for _ in ()).throw(error)
        status,body=probe_llm_api.request_json("https://example.test/v1/models","secret")
        assert status==403 and "Access denied" in body["error"]["message"] and body["_http"]["server"]=="cloudflare"
    finally:
        probe_llm_api.urlopen=original
