from __future__ import annotations

from pc_mcmc_cigp.agent_backend.models import (
    ChemicalSpeciesSpec, ElementaryStepSpec, ExperimentalVariable, MechanismSpec,
    Observable, ProjectStage, ReactionProjectSpec,
)


def project_from_dict(payload: dict) -> ReactionProjectSpec:
    return ReactionProjectSpec(
        project_id=payload["project_id"], title=payload["title"], objective=payload["objective"],
        raw_user_request=payload.get("raw_user_request", ""),
        reactants=[ChemicalSpeciesSpec(**item) for item in payload.get("reactants", [])],
        known_products=[ChemicalSpeciesSpec(**item) for item in payload.get("known_products", [])],
        suspected_intermediates=[ChemicalSpeciesSpec(**item) for item in payload.get("suspected_intermediates", [])],
        catalysts=[ChemicalSpeciesSpec(**item) for item in payload.get("catalysts", [])],
        solvents=[ChemicalSpeciesSpec(**item) for item in payload.get("solvents", [])],
        controllable_variables=[ExperimentalVariable(**item) for item in payload.get("controllable_variables", [])],
        observed_variables=[Observable(**item) for item in payload.get("observed_variables", [])],
        known_conditions=payload.get("known_conditions", {}), constraints=payload.get("constraints", []),
        candidate_reaction_families=payload.get("candidate_reaction_families", []), uncertainties=payload.get("uncertainties", []),
        missing_information=payload.get("missing_information", []), stage=ProjectStage(payload.get("stage", "intake")),
        schema_version=payload.get("schema_version", "1.0"),
    )


def mechanism_from_dict(payload: dict) -> MechanismSpec:
    return MechanismSpec(
        mechanism_id=payload["mechanism_id"], project_id=payload["project_id"],
        species=tuple(ChemicalSpeciesSpec(**item) for item in payload.get("species", [])),
        steps=tuple(ElementaryStepSpec(**item) for item in payload.get("steps", [])),
        approved=bool(payload.get("approved", False)), schema_version=payload.get("schema_version", "1.0"),
    )
