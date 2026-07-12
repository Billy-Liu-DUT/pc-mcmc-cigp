from __future__ import annotations

import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request_json(url: str, key: str, payload: dict | None = None) -> tuple[int, dict]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        method="GET" if payload is None else "POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        try:
            body = json.loads(exc.read().decode("utf-8"))
        except Exception:
            body = {"error": {"message": str(exc)}}
        return exc.code, body
    except URLError as exc:
        return 0, {"error": {"message": str(exc.reason)}}


def error_message(body: dict) -> str:
    error = body.get("error", body)
    if isinstance(error, dict):
        return str(error.get("message", error.get("type", "unknown error")))[:240]
    return str(error)[:240]


def main() -> int:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    base = os.environ.get("OPENAI_BASE_URL", "").strip().rstrip("/")
    model = os.environ.get("OPENAI_MODEL", "gpt-5.5").strip()
    if not key or not base:
        print("Set OPENAI_API_KEY and OPENAI_BASE_URL in the current shell. Nothing was sent.")
        return 2

    print(f"Testing configured endpoint with model={model!r}; credentials will not be printed.")
    status, body = request_json(f"{base}/models", key)
    models = [item.get("id") for item in body.get("data", []) if isinstance(item, dict)]
    print(json.dumps({"test": "models", "status": status, "models": models[:20], "error": None if status == 200 else error_message(body)}, ensure_ascii=False))

    responses_payload = {"model": model, "input": "Reply with exactly: API_OK", "max_output_tokens": 16}
    status, body = request_json(f"{base}/responses", key, responses_payload)
    output_text = body.get("output_text")
    print(json.dumps({"test": "responses", "status": status, "model": body.get("model"), "output": output_text, "error": None if status == 200 else error_message(body)}, ensure_ascii=False))
    if status == 200:
        return 0

    chat_payload = {"model": model, "messages": [{"role": "user", "content": "Reply with exactly: API_OK"}], "max_tokens": 16}
    status, body = request_json(f"{base}/chat/completions", key, chat_payload)
    choices = body.get("choices", [])
    output = choices[0].get("message", {}).get("content") if choices else None
    print(json.dumps({"test": "chat_completions", "status": status, "model": body.get("model"), "output": output, "error": None if status == 200 else error_message(body)}, ensure_ascii=False))
    return 0 if status == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
