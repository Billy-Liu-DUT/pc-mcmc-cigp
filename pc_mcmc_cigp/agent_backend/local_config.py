from __future__ import annotations

import os
from pathlib import Path


def load_local_env(path: str | Path | None = None) -> Path | None:
    """Load missing provider variables from an ignored local file without logging values."""
    candidates = (
        [Path(path)]
        if path
        else [Path.cwd() / ".env.local", Path(__file__).resolve().parents[2] / ".env.local"]
    )
    selected = next((item for item in candidates if item.is_file()), None)
    if selected is None:
        return None
    for line in selected.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key in {"OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"} and value:
            os.environ.setdefault(key, value)
    return selected
