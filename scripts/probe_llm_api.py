from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pc_mcmc_cigp.agent_backend.local_config import load_local_env


def request_json(url: str, key: str, payload: dict | None = None) -> tuple[int, dict]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        method="GET" if payload is None else "POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) CIGP-Agent/0.1",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
        except Exception:
            compact = " ".join(raw.split())[:300]
            body = {"error": {"message": compact or str(exc)}}
        body["_http"] = {
            "server": exc.headers.get("Server"),
            "content_type": exc.headers.get("Content-Type"),
            "request_id": exc.headers.get("x-request-id") or exc.headers.get("cf-ray"),
        }
        return exc.code, body
    except URLError as exc:
        return 0, {"error": {"message": str(exc.reason)}}


def error_message(body: dict) -> str:
    error = body.get("error", body)
    if isinstance(error, dict):
        return str(error.get("message", error.get("type", "unknown error")))[:240]
    return str(error)[:240]


def http_metadata(body: dict) -> dict:
    return body.get("_http", {}) if isinstance(body, dict) else {}


def response_output_text(body: dict) -> str | None:
    if isinstance(body.get("output_text"), str):
        return body["output_text"]
    texts = []
    for item in body.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                texts.append(content["text"])
    return "\n".join(texts) if texts else None


def main() -> int:
    load_local_env(ROOT / ".env.local")
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    base = os.environ.get("OPENAI_BASE_URL", "").strip().rstrip("/")
    model = os.environ.get("OPENAI_MODEL", "gpt-5.5").strip()
    if not key or not base:
        print("Set OPENAI_API_KEY and OPENAI_BASE_URL in the current shell. Nothing was sent.")
        return 2

    print(f"Testing configured endpoint with model={model!r}; credentials will not be printed.")
    status, body = request_json(f"{base}/models", key)
    models = [item.get("id") for item in body.get("data", []) if isinstance(item, dict)]
    print(json.dumps({"test": "models", "status": status, "models": models[:20], "error": None if status == 200 else error_message(body), "http": http_metadata(body)}, ensure_ascii=False))

    responses_payload = {"model": model, "input": "Reply with exactly: API_OK", "max_output_tokens": 16}
    status, body = request_json(f"{base}/responses", key, responses_payload)
    output_text = response_output_text(body)
    print(json.dumps({"test": "responses", "status": status, "model": body.get("model"), "output": output_text, "error": None if status == 200 else error_message(body), "http": http_metadata(body)}, ensure_ascii=False))
    if status == 200:
        return 0

    chat_payload = {"model": model, "messages": [{"role": "user", "content": "Reply with exactly: API_OK"}], "max_tokens": 16}
    status, body = request_json(f"{base}/chat/completions", key, chat_payload)
    choices = body.get("choices", [])
    output = choices[0].get("message", {}).get("content") if choices else None
    print(json.dumps({"test": "chat_completions", "status": status, "model": body.get("model"), "output": output, "error": None if status == 200 else error_message(body), "http": http_metadata(body)}, ensure_ascii=False))
    return 0 if status == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
