from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pc_mcmc_cigp.agent_backend.llm_client import CompatibleResponsesClient


class ChatSessionStore:
    def __init__(self, project_root: str | Path) -> None:
        self.root = Path(project_root) / ".chat_sessions"
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self) -> str:
        session_id = f"chat_{uuid4().hex}"
        self._path(session_id).touch()
        return session_id

    def append(self, session_id: str, role: str, content: str, metadata: dict | None = None) -> None:
        if role not in {"user", "assistant"} or not content.strip():
            raise ValueError("chat messages require a valid role and non-empty content")
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), "role": role, "content": content, "metadata": metadata or {}}
        with self._path(session_id).open("a", encoding="utf-8", newline="") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read(self, session_id: str, limit: int = 50) -> list[dict]:
        path = self._path(session_id)
        if not path.exists():
            raise FileNotFoundError("unknown chat session")
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return records[-limit:]

    def _path(self, session_id: str) -> Path:
        if not session_id.startswith("chat_") or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for char in session_id):
            raise ValueError("invalid chat session id")
        return self.root / f"{session_id}.jsonl"


class ConversationService:
    def __init__(self, project_root: str | Path) -> None:
        self.store = ChatSessionStore(project_root)

    def send(self, message: str, session_id: str | None = None, project_context: dict | None = None) -> dict:
        session_id = session_id or self.store.create()
        self.store.append(session_id, "user", message)
        history = [{"role": row["role"], "content": row["content"]} for row in self.store.read(session_id)]
        result = CompatibleResponsesClient.from_env().chat(history, project_context)
        self.store.append(session_id, "assistant", result["reply"], {key: value for key, value in result.items() if key != "reply"})
        return {"session_id": session_id, **result}
