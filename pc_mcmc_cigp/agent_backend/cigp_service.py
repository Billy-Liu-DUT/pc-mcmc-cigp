from __future__ import annotations

from uuid import uuid4

import numpy as np

from pc_mcmc_cigp.acquisition import AcquisitionFactory
from pc_mcmc_cigp.agent_backend.models import CIGPReport, MCMCSummary, TemplateScore
from pc_mcmc_cigp.cigp import CIGPConfig, CIGPRegressor
from pc_mcmc_cigp.kinetics import TemplateRegistry, create_kinetic_template
from pc_mcmc_cigp.agent_backend.network_kinetics import compile_posterior_kinetics


class CIGPService:
    def rank_templates(self, X: np.ndarray, y: np.ndarray, candidates: list[str]) -> list[TemplateScore]:
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float).ravel(); scores = []
        for name in candidates:
            try:
                model = create_kinetic_template(name)
                if X.ndim != 2 or X.shape[1] != len(model.input_names):
                    scores.append(TemplateScore(name, False, None, f"requires inputs {model.input_names}")); continue
                fitted = CIGPRegressor(model, CIGPConfig(optimize_hyperparameters=False)).fit(X, y)
                mean, _ = fitted.predict(X); rmse = float(np.sqrt(np.mean((mean.ravel() - y) ** 2)))
                scores.append(TemplateScore(name, True, rmse, "compatible and numerically fitted"))
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                scores.append(TemplateScore(name, False, None, str(exc)))
        return sorted(scores, key=lambda item: (not item.compatible, np.inf if item.rmse is None else item.rmse))

    def fit_and_recommend(
        self, mcmc: MCMCSummary, template_name: str, X: np.ndarray, y: np.ndarray,
        bounds: dict[str, tuple[float, float]], *, objective: str = "maximize_yield", n_candidates: int = 512,
        allow_unconverged_mcmc: bool = False, random_state: int = 0,
    ) -> CIGPReport:
        if mcmc is not None and not mcmc.converged and not allow_unconverged_mcmc:
            raise PermissionError("CIGP optimization requires converged PC-MCMC or an explicit override")
        physics = create_kinetic_template(template_name)
        return self.fit_model_and_recommend(physics, X, y, bounds, objective=objective, n_candidates=n_candidates, random_state=random_state, template_name=template_name)

    def fit_compiled_and_recommend(
        self, mechanism, mcmc: MCMCSummary, target_species: str, X, y, bounds,
        *, inclusion_threshold: float = 0.5, objective: str = "maximize_yield", n_candidates: int = 512,
        allow_unconverged_mcmc: bool = False, random_state: int = 0,
    ) -> CIGPReport:
        if not mcmc.converged and not allow_unconverged_mcmc:
            raise PermissionError("compiled CIGP requires converged PC-MCMC or an explicit override")
        physics = compile_posterior_kinetics(mechanism, mcmc, target_species, inclusion_threshold=inclusion_threshold)
        return self.fit_model_and_recommend(physics, X, y, bounds, objective=objective, n_candidates=n_candidates, random_state=random_state, template_name=f"mcmc_compiled:{mechanism.mechanism_id}")

    def fit_model_and_recommend(
        self, physics, X, y, bounds, *, objective="maximize_yield", n_candidates=512, random_state=0,
        template_name="custom",
    ) -> CIGPReport:
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float).ravel()
        if tuple(bounds) != tuple(physics.input_names):
            raise ValueError(f"bounds must be ordered exactly as {physics.input_names}")
        lower = np.asarray([bounds[name][0] for name in physics.input_names]); upper = np.asarray([bounds[name][1] for name in physics.input_names])
        if np.any(upper <= lower): raise ValueError("every upper bound must exceed its lower bound")
        model = CIGPRegressor(physics, CIGPConfig(random_state=random_state)).fit(X, y)
        rng = np.random.default_rng(random_state); candidates = lower + rng.random((n_candidates, len(lower))) * (upper - lower)
        acquisition = AcquisitionFactory.create("EI", xi=0.01)
        scores = acquisition.compute(model, candidates, y_best=float(np.max(y)))
        index = int(np.argmax(scores)); mean, variance = model.predict(candidates[index:index+1])
        recommendation = {
            "conditions": {name: float(value) for name, value in zip(physics.input_names, candidates[index])},
            "predicted_mean": float(mean[0, 0]), "predicted_std": float(np.sqrt(variance[0, 0])),
            "acquisition": "EI", "acquisition_score": float(scores[index]),
            "is_extrapolation": bool(np.any(candidates[index] < np.min(X, axis=0)) or np.any(candidates[index] > np.max(X, axis=0))),
        }
        warnings = ("Recommendation lies outside the observed design envelope",) if recommendation["is_extrapolation"] else ()
        return CIGPReport(f"cigp_{uuid4().hex[:12]}", template_name, len(X), objective, float(np.max(y)), recommendation, warnings)


def available_template_contracts() -> list[dict]:
    return [TemplateRegistry.describe(name) for name in TemplateRegistry.names()]
