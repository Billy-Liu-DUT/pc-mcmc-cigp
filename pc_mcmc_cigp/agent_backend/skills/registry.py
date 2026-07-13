from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SkillDefinition:
    name: str
    prompt_file: str
    schema_file: str
    stages: tuple[str, ...]
    description: str


class SkillRegistry:
    DEFINITIONS = (
        SkillDefinition(
            "reaction_intake",
            "reaction_intake.md",
            "reaction_intake.schema.json",
            ("intake", "needs_clarification"),
            "Turn a reaction request into a reviewable project draft.",
        ),
        SkillDefinition(
            "experiment_design",
            "experiment_design.md",
            "experiment_design.schema.json",
            ("experiment_plan_ready", "needs_more_data"),
            "Design discriminating kinetic experiments.",
        ),
        SkillDefinition(
            "data_template",
            "data_template.md",
            "data_template.schema.json",
            ("waiting_for_data",),
            "Explain and generate the upload data contract.",
        ),
        SkillDefinition(
            "data_quality",
            "data_quality.md",
            "data_quality.schema.json",
            ("data_mapping", "data_validation"),
            "Explain deterministic data validation and identifiability findings.",
        ),
        SkillDefinition(
            "mechanism_hypothesis",
            "mechanism_hypothesis.md",
            "mechanism_hypothesis.schema.json",
            ("mechanism_proposal", "waiting_for_mechanism_approval"),
            "Propose finite, balanced candidate elementary steps.",
        ),
        SkillDefinition(
            "mcmc_interpretation",
            "mcmc_interpretation.md",
            "mcmc_interpretation.schema.json",
            ("mcmc_running", "mcmc_review"),
            "Interpret PC-MCMC posterior diagnostics without overclaiming.",
        ),
        SkillDefinition(
            "cigp_optimization",
            "cigp_optimization.md",
            "cigp_optimization.schema.json",
            (
                "cigp_model_compilation",
                "cigp_fitting",
                "cigp_running",
                "optimization_ready",
                "waiting_for_next_experiment",
            ),
            "Interpret CIGP recommendations and uncertainty.",
        ),
    )

    def __init__(self) -> None:
        self._by_name = {item.name: item for item in self.DEFINITIONS}

    def get(self, name: str) -> SkillDefinition:
        if name not in self._by_name:
            raise KeyError(f"unknown prompt skill: {name}")
        return self._by_name[name]

    def route(self, stage: str | None, message: str) -> SkillDefinition:
        stage = (stage or "intake").lower()
        text = message.lower()
        if any(term in text for term in ("csv", "格式", "上传", "表格", "模板", "template")) and stage in {
            "waiting_for_data",
            "experiment_plan_ready",
            "needs_more_data",
        }:
            return self.get("data_template")
        for definition in self.DEFINITIONS:
            if stage in definition.stages:
                return definition
        return self.get("reaction_intake")

    def list(self) -> tuple[SkillDefinition, ...]:
        return self.DEFINITIONS
