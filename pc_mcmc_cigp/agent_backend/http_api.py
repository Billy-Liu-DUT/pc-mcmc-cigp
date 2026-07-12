from __future__ import annotations

import json
import mimetypes
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from pc_mcmc_cigp.agent_backend.cigp_service import available_template_contracts
from pc_mcmc_cigp.agent_backend.codecs import mechanism_from_dict, project_from_dict
from pc_mcmc_cigp.agent_backend.frontend import FrontendReadModel, api_placeholder_status
from pc_mcmc_cigp.agent_backend.workflow import ReactionAgentWorkflow
from pc_mcmc_cigp.agent_backend.models import MCMCSummary
from pc_mcmc_cigp.agent_backend.llm_client import CompatibleResponsesClient, LLMConfigurationError, LLMRequestError, client_status
from pc_mcmc_cigp.agent_backend.chat import ConversationService


class ReactionAPI:
    def __init__(self, project_root: str | Path, static_root: str | Path) -> None:
        self.workflow = ReactionAgentWorkflow(project_root)
        self.frontend = FrontendReadModel(self.workflow.store)
        self.static_root = Path(static_root).resolve()
        self.conversation = ConversationService(project_root)

    def dispatch(self, method: str, path: str, payload: dict | None = None) -> tuple[int, dict | bytes, str]:
        payload = payload or {}; parts = [item for item in path.strip("/").split("/") if item]
        if path == "/api/health" and method == "GET":
            return 200, {"status": "ok", "api": api_placeholder_status(), "service": "pc-mcmc-cigp"}, "application/json"
        if path == "/api/templates" and method == "GET":
            return 200, {"templates": available_template_contracts()}, "application/json"
        if path == "/api/llm/status" and method == "GET":
            return 200, client_status(), "application/json"
        if path == "/api/llm/parse-project" and method == "POST":
            project=CompatibleResponsesClient.from_env().parse_project(payload["user_request"],payload["project_id"])
            return 200,{"project":project.to_dict()},"application/json"
        if path == "/api/llm/propose-mechanism" and method == "POST":
            mechanism=CompatibleResponsesClient.from_env().propose_mechanism(payload["project"],payload.get("dataset_summary"))
            return 200,{"mechanism":asdict(mechanism),"graph":self.frontend.mechanism_graph(mechanism)},"application/json"
        if path == "/api/chat" and method == "POST":
            result=self.conversation.send(payload["message"],payload.get("session_id"),payload.get("project_context"))
            return 200,result,"application/json"
        if len(parts)==3 and parts[:2]==["api","chat"] and method=="GET":
            return 200,{"session_id":parts[2],"messages":self.conversation.store.read(parts[2])},"application/json"
        if path == "/api/projects" and method == "POST":
            project = project_from_dict(payload); self.workflow.create_project(project)
            return 201, {"project": project.to_dict(), "dashboard": asdict(self.frontend.dashboard(project.project_id))}, "application/json"
        if len(parts) == 4 and parts[:2] == ["api", "projects"] and parts[3] == "dashboard" and method == "GET":
            return 200, asdict(self.frontend.dashboard(parts[2])), "application/json"
        if len(parts) == 4 and parts[:2] == ["api", "projects"] and parts[3] == "experiment-plan" and method == "POST":
            project = project_from_dict(self.workflow.store.load(parts[2])); requests = self.workflow.prepare_experiment_plan(project)
            return 200, {"project": project.to_dict(), "requests": [asdict(item) for item in requests]}, "application/json"
        if len(parts) == 5 and parts[:2] == ["api", "projects"] and parts[3:] == ["datasets", "validate"] and method == "POST":
            project = project_from_dict(self.workflow.store.load(parts[2])); incoming = self.workflow.store.root / parts[2] / "incoming.csv"
            incoming.write_text(payload.get("csv_text", ""), encoding="utf-8", newline="")
            report, rows = self.workflow.ingest_dataset(project, incoming)
            return (200 if report.valid else 422), {"project": project.to_dict(), "report": asdict(report), "row_count": len(rows)}, "application/json"
        if path == "/api/mechanisms/compile" and method == "POST":
            spec = mechanism_from_dict(payload); compiled = self.workflow.algorithms.compile_mechanism(spec)
            return (200 if compiled.report.valid else 422), {"report": asdict(compiled.report), "graph": self.frontend.mechanism_graph(spec)}, "application/json"
        if len(parts) == 5 and parts[:2] == ["api", "projects"] and parts[3:] == ["mechanisms", "register"] and method == "POST":
            project = project_from_dict(self.workflow.store.load(parts[2])); spec = mechanism_from_dict(payload)
            report = self.workflow.register_mechanism(project, spec)
            return (200 if report.valid else 422), {"project": project.to_dict(), "report": asdict(report)}, "application/json"
        if len(parts) == 5 and parts[:2] == ["api", "projects"] and parts[3:] == ["mechanisms", "approve"] and method == "POST":
            project = project_from_dict(self.workflow.store.load(parts[2])); approved = self.workflow.approve_mechanism(project, mechanism_from_dict(payload))
            return 200, {"project": project.to_dict(), "mechanism": asdict(approved)}, "application/json"
        if len(parts) == 5 and parts[:2] == ["api", "projects"] and parts[3:] == ["mcmc", "run"] and method == "POST":
            project = project_from_dict(self.workflow.store.load(parts[2])); spec = mechanism_from_dict(payload["mechanism"])
            dataset_path = self.workflow.store.root / parts[2] / "datasets" / payload.get("dataset_version", "v001.json")
            rows = json.loads(dataset_path.read_text(encoding="utf-8")); config = payload.get("config", {})
            summary = self.workflow.run_mcmc(project, spec, rows, sources=payload["sources"], targets=payload["targets"], **config)
            return 200, {"project": project.to_dict(), "summary": asdict(summary)}, "application/json"
        if len(parts) == 5 and parts[:2] == ["api", "projects"] and parts[3:] == ["cigp", "run"] and method == "POST":
            project = project_from_dict(self.workflow.store.load(parts[2])); mcmc = MCMCSummary(**payload["mcmc_summary"])
            report = self.workflow.run_cigp(project, mcmc, payload["template_name"], payload["X"], payload["y"], payload["bounds"], **payload.get("config", {}))
            return 200, {"project": project.to_dict(), "report": asdict(report)}, "application/json"
        return self._static(path) if method == "GET" and not path.startswith("/api/") else (404, {"error": "route_not_found", "path": path}, "application/json")

    def _static(self, path: str) -> tuple[int, bytes | dict, str]:
        relative = "index.html" if path in {"/", ""} else path.lstrip("/")
        candidate = (self.static_root / relative).resolve()
        if not str(candidate).startswith(str(self.static_root)) or not candidate.is_file():
            candidate = self.static_root / "index.html"
        if not candidate.is_file(): return 404, {"error": "frontend_not_found"}, "application/json"
        return 200, candidate.read_bytes(), mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"


def make_handler(api: ReactionAPI):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self): self._handle("GET")
        def do_POST(self): self._handle("POST")
        def _handle(self, method):
            try:
                length = int(self.headers.get("Content-Length", "0")); raw = self.rfile.read(length) if length else b""
                payload = json.loads(raw.decode("utf-8")) if raw else {}
                status, body, content_type = api.dispatch(method, urlparse(self.path).path, payload)
            except (ValueError, KeyError, FileExistsError, FileNotFoundError, PermissionError, LLMConfigurationError, LLMRequestError) as exc:
                status, body, content_type = 400, {"error": type(exc).__name__, "message": str(exc)}, "application/json"
            encoded = json.dumps(body, ensure_ascii=False, default=str).encode("utf-8") if isinstance(body, dict) else body
            self.send_response(status); self.send_header("Content-Type", f"{content_type}; charset=utf-8"); self.send_header("Content-Length", str(len(encoded))); self.end_headers(); self.wfile.write(encoded)
        def log_message(self, format, *args): return
    return Handler


def serve(host="127.0.0.1", port=8765, project_root="projects", static_root="web"):
    api = ReactionAPI(project_root, static_root); server = ThreadingHTTPServer((host, port), make_handler(api))
    print(f"CIGP workbench: http://{host}:{port}"); server.serve_forever()
