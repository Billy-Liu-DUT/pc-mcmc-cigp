from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pc_mcmc_cigp.agent_backend.codecs import mechanism_from_dict, project_from_dict
from pc_mcmc_cigp.agent_backend.skills import SchemaValidationError, SkillRuntime
from pc_mcmc_cigp.agent_backend.local_config import load_local_env


class LLMConfigurationError(RuntimeError): pass
class LLMRequestError(RuntimeError): pass
Transport = Callable[[str, str, dict], dict]
_PROVIDER_SEMAPHORE = threading.BoundedSemaphore(1)


@dataclass(frozen=True)
class LLMProviderConfig:
    base_url: str
    model: str
    api_key: str
    provider_name: str = "openai-compatible"

    @classmethod
    def from_env(cls):
        load_local_env()
        key=os.environ.get("OPENAI_API_KEY","").strip(); base=os.environ.get("OPENAI_BASE_URL","").strip().rstrip("/"); model=os.environ.get("OPENAI_MODEL","").strip()
        if not key or not base or not model: raise LLMConfigurationError("OPENAI_API_KEY, OPENAI_BASE_URL and OPENAI_MODEL must all be set")
        if not base.startswith("https://"): raise LLMConfigurationError("OPENAI_BASE_URL must use HTTPS")
        return cls(base,model,key)

    def public_status(self): return {"configured":True,"base_url":self.base_url,"model":self.model,"provider":self.provider_name}


class CompatibleResponsesClient:
    def __init__(self,config:LLMProviderConfig,transport:Transport|None=None,skill_runtime:SkillRuntime|None=None):
        self.config=config; self.transport=transport or self._http_transport; self.skills=skill_runtime or SkillRuntime()
    @classmethod
    def from_env(cls): return cls(LLMProviderConfig.from_env())

    def parse_project(self,user_request:str,project_id:str):
        if not user_request.strip(): raise ValueError("user_request must not be empty")
        prompt=f"""Return one JSON object for this reaction project. Required keys: project_id,title,objective,raw_user_request,reactants,known_products,suspected_intermediates,catalysts,solvents,controllable_variables,observed_variables,known_conditions,constraints,candidate_reaction_families,uncertainties,missing_information. Species contain name, formula element-count object, charge, role, status. Guesses use status llm_candidate. project_id={project_id!r}. Request: {user_request}"""
        payload=self._json_response("Structure reaction-kinetics projects. Return valid JSON only.",prompt); payload["project_id"]=project_id; payload["raw_user_request"]=user_request
        return project_from_dict(payload)

    def propose_mechanism(self,project_payload:dict,dataset_summary:dict|None=None):
        prompt=f"""Return one JSON candidate mechanism with mechanism_id,project_id,species,steps,approved=false. Steps require step_id,reactants,products,catalysts,rate_law_family,reversible,prior_probability,evidence,status=llm_candidate. Allowed rate laws: mass_action,power_law,arrhenius,reversible,saturation,inhibition. Conserve formulas. Project: {json.dumps(project_payload,ensure_ascii=False)} Dataset: {json.dumps(dataset_summary or {},ensure_ascii=False)}"""
        payload=self._json_response("Propose testable kinetic hypotheses. Return valid JSON only.",prompt); payload["project_id"]=project_payload["project_id"]; payload["approved"]=False
        return mechanism_from_dict(payload)

    def chat(self, messages: list[dict], project_context: dict | None = None) -> dict:
        history = messages[-20:]; context = dict(project_context or {})
        last_message = next((str(item.get("content", "")) for item in reversed(history) if item.get("role") == "user"), "")
        skill = self.skills.route(context.get("stage"), last_message)
        instructions, prompt = self.skills.render(skill, context, history)
        payload = self._normalize_skill_payload(skill.name, self._json_response(instructions, prompt), context)
        repaired = False
        try:
            self.skills.validate(skill, payload)
        except SchemaValidationError as exc:
            repair_prompt = (
                f"Repair the JSON to match the schema without adding scientific claims. Errors: {exc.errors}. "
                f"Invalid JSON: {json.dumps(payload, ensure_ascii=False)}\n{prompt}"
            )
            payload = self._normalize_skill_payload(skill.name, self._json_response(instructions, repair_prompt), context)
            self.skills.validate(skill, payload); repaired = True
        payload.setdefault("reply", "")
        payload.setdefault("intent", "general")
        payload.setdefault("missing_information", [])
        payload.setdefault("suggested_actions", [])
        payload.setdefault("project_draft", None)
        payload["active_skill"] = skill.name
        payload["schema_repaired"] = repaired
        return payload

    @staticmethod
    def _normalize_skill_payload(skill_name: str, payload: dict, context: dict) -> dict:
        """Fill compatibility metadata while leaving scientific artifacts schema-checked."""
        payload = dict(payload)
        defaults = {
            "reply": "", "intent": "clarify", "missing_information": [], "suggested_actions": [],
            "requires_user_confirmation": False, "project_draft": None,
            "workflow_mode": context.get("workflow_mode", "coupled"),
        }
        stage_defaults = {
            "reaction_intake": "intake", "experiment_design": "experiment_plan_ready",
            "data_template": "waiting_for_data", "data_quality": "data_validation",
            "mechanism_hypothesis": "mechanism_proposal", "mcmc_interpretation": "mcmc_review",
            "cigp_optimization": "optimization_ready",
        }
        defaults["current_stage"] = context.get("stage", stage_defaults[skill_name])
        extras = {
            "reaction_intake": {"known_facts": [], "hypotheses": []},
            "experiment_design": {"experiment_plan": []},
            "data_template": {"data_contract": {}},
            "data_quality": {"decision": "request_more_data", "blocking_issues": [], "warnings": [], "identifiability_risks": []},
            "mechanism_hypothesis": {"mechanism_draft": {}, "assumptions": [], "rejected_candidates": []},
            "mcmc_interpretation": {"decision": "collect_data", "evidence": [], "limitations": [], "proposed_adjustments": []},
            "cigp_optimization": {"template_source": "predefined", "decision": "propose_experiment", "recommendations": [], "warnings": []},
        }
        for key, value in {**defaults, **extras[skill_name]}.items(): payload.setdefault(key, value)
        return payload

    def _json_response(self,instructions,prompt):
        with _PROVIDER_SEMAPHORE:
            body=self.transport(f"{self.config.base_url}/responses",self.config.api_key,{"model":self.config.model,"instructions":instructions,"input":prompt,"max_output_tokens":4096})
        text=self._extract_text(body); match=re.search(r"\{.*\}",text,re.DOTALL)
        if not match: raise LLMRequestError("model response did not contain a JSON object")
        try: return json.loads(match.group())
        except json.JSONDecodeError as exc: raise LLMRequestError(f"model returned invalid JSON: {exc.msg}") from exc

    @staticmethod
    def _extract_text(body):
        if isinstance(body.get("output_text"),str): return body["output_text"]
        texts=[content["text"] for item in body.get("output",[]) if isinstance(item,dict) for content in item.get("content",[]) if isinstance(content,dict) and isinstance(content.get("text"),str)]
        if not texts: raise LLMRequestError("Responses payload contained no output text")
        return "\n".join(texts)

    @staticmethod
    def _http_transport(url,key,payload):
        request=Request(url,data=json.dumps(payload).encode(),method="POST",headers={
            "Authorization":f"Bearer {key}","Content-Type":"application/json","Accept":"application/json",
            "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) CIGP-Agent/0.1",
        })
        try:
            with urlopen(request,timeout=90) as response: return json.loads(response.read().decode())
        except HTTPError as exc:
            try: message=json.loads(exc.read().decode()).get("error",{}).get("message",str(exc))
            except Exception: message=str(exc)
            raise LLMRequestError(f"provider returned HTTP {exc.code}: {message}") from exc
        except URLError as exc: raise LLMRequestError(f"provider connection failed: {exc.reason}") from exc


def client_status():
    try:
        status=LLMProviderConfig.from_env().public_status(); status["max_concurrency"]=1; return status
    except LLMConfigurationError as exc:
        return {"configured":False,"base_url":None,"model":None,"provider":None,"max_concurrency":1,"message":str(exc)}
