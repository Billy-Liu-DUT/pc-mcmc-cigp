from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pc_mcmc_cigp.agent_backend.models import ProjectStage, ReactionProjectSpec


class ProjectStore:
    """Versioned, append-only artifact store for a reaction project."""

    FOLDERS = ("datasets", "experiment_requests", "mechanisms", "mcmc_runs", "cigp_runs", "reports")

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def create(self, project: ReactionProjectSpec) -> Path:
        project_dir = self.root / project.project_id
        if project_dir.exists():
            raise FileExistsError(f"project {project.project_id!r} already exists")
        project_dir.mkdir(parents=True)
        for folder in self.FOLDERS:
            (project_dir / folder).mkdir()
        self._write_json(project_dir / "project.json", project.to_dict())
        self.append_event(project.project_id, "project_created", {"stage": project.stage.value})
        return project_dir

    def load(self, project_id: str) -> dict[str, Any]:
        return json.loads((self.root / project_id / "project.json").read_text(encoding="utf-8"))

    def save_project(self, project: ReactionProjectSpec) -> None:
        self._ensure_project(project.project_id)
        self._write_json(self.root / project.project_id / "project.json", project.to_dict())

    def transition(self, project: ReactionProjectSpec, target: ProjectStage, note: str = "") -> None:
        previous = project.stage
        project.transition(target)
        self.save_project(project)
        self.append_event(project.project_id, "stage_transition", {"from": previous.value, "to": target.value, "note": note})

    def next_version_path(self, project_id: str, category: str, suffix: str) -> Path:
        self._ensure_project(project_id)
        if category not in self.FOLDERS:
            raise ValueError(f"unknown artifact category: {category}")
        folder = self.root / project_id / category
        existing = sorted(folder.glob(f"v*.{suffix.lstrip('.')}"))
        return folder / f"v{len(existing) + 1:03d}.{suffix.lstrip('.')}"

    def save_artifact(self, project_id: str, category: str, payload: Any, suffix: str = "json") -> Path:
        path = self.next_version_path(project_id, category, suffix)
        if suffix == "json":
            if hasattr(payload, "__dataclass_fields__"):
                payload = asdict(payload)
            self._write_json(path, payload)
        elif isinstance(payload, bytes):
            path.write_bytes(payload)
        else:
            path.write_text(str(payload), encoding="utf-8", newline="")
        self.append_event(project_id, "artifact_saved", {"category": category, "path": str(path.name)})
        return path

    def append_event(self, project_id: str, event: str, details: dict[str, Any]) -> None:
        project_dir = self.root / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event, "details": details}
        with (project_dir / "events.jsonl").open("a", encoding="utf-8", newline="") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _ensure_project(self, project_id: str) -> None:
        if not (self.root / project_id / "project.json").exists():
            raise FileNotFoundError(f"unknown project: {project_id}")

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8", newline="")
