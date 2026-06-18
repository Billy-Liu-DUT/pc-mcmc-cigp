from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseAcquisition(ABC):
    def __init__(self, **params) -> None:
        self.params = params

    @abstractmethod
    def compute(self, model, X_candidates: np.ndarray, y_best: float | None = None) -> np.ndarray:
        raise NotImplementedError


class ExpectedImprovement(BaseAcquisition):
    def compute(self, model, X_candidates: np.ndarray, y_best: float | None = None) -> np.ndarray:
        mu, var = model.predict(X_candidates)
        if y_best is None:
            y_best = float(np.max(mu))
        sigma = np.sqrt(np.maximum(var, 1e-12))
        xi = self.params.get("xi", 0.01)
        improvement = mu - y_best - xi
        z = improvement / sigma
        scores = improvement * _normal_cdf(z) + sigma * _normal_pdf(z)
        scores[sigma < 1e-9] = 0.0
        return scores.ravel()


class GradientWeightedUncertainty(BaseAcquisition):
    def compute(self, model, X_candidates: np.ndarray, y_best: float | None = None) -> np.ndarray:
        _, var = model.predict(X_candidates)
        gradients = model.physics_model.compute_gradients_W(X_candidates, model.W_)
        sensitivity = np.linalg.norm(gradients, axis=1)
        return np.sqrt(np.maximum(var.ravel(), 0.0)) * sensitivity


class DiscrepancyHunter(BaseAcquisition):
    def compute(self, model, X_candidates: np.ndarray, y_best: float | None = None) -> np.ndarray:
        mu, var = model.predict_residual(X_candidates)
        beta = self.params.get("beta", 1.0)
        return np.abs(mu.ravel()) + beta * np.sqrt(np.maximum(var.ravel(), 0.0))


class PhysConstrainedEI(BaseAcquisition):
    def compute(self, model, X_candidates: np.ndarray, y_best: float | None = None) -> np.ndarray:
        base = ExpectedImprovement(**self.params).compute(model, X_candidates, y_best)
        phys = model.physics_model.compute_mean(X_candidates, model.W_).ravel()
        threshold = self.params.get("threshold", 0.1)
        sharpness = self.params.get("sharpness", 10.0)
        weight = 1.0 / (1.0 + np.exp(-sharpness * (phys - threshold)))
        return base * weight


class AcquisitionFactory:
    _REGISTRY = {
        "EI": ExpectedImprovement,
        "GWU": GradientWeightedUncertainty,
        "DH": DiscrepancyHunter,
        "PC_EI": PhysConstrainedEI,
    }

    @classmethod
    def create(cls, name: str, **params) -> BaseAcquisition:
        try:
            return cls._REGISTRY[name](**params)
        except KeyError as exc:
            valid = ", ".join(sorted(cls._REGISTRY))
            raise ValueError(f"Unknown acquisition function {name!r}. Valid options: {valid}") from exc


def _normal_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(_erf_approx)(x / np.sqrt(2.0)))


def _erf_approx(x: float) -> float:
    sign = 1.0 if x >= 0 else -1.0
    x = abs(float(x))
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return sign * y
