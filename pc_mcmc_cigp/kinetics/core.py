from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

try:
    from scipy.integrate import solve_ivp
except ImportError:  # pragma: no cover
    solve_ivp = None


@dataclass(frozen=True)
class KineticParameter:
    name: str
    default: float
    bounds: tuple[float, float] = (-np.inf, np.inf)
    unit: str = ""
    transform: str = "identity"


@dataclass(frozen=True)
class InputTransform:
    minimum: np.ndarray
    maximum: np.ndarray

    def __post_init__(self) -> None:
        lo = np.asarray(self.minimum, dtype=float)
        hi = np.asarray(self.maximum, dtype=float)
        if lo.shape != hi.shape or np.any(hi <= lo):
            raise ValueError("minimum and maximum must have equal shapes and maximum > minimum")
        object.__setattr__(self, "minimum", lo)
        object.__setattr__(self, "maximum", hi)

    @classmethod
    def from_legacy(cls, scaler: dict) -> "InputTransform":
        return cls(np.asarray(scaler["min"], dtype=float), np.asarray(scaler["max"], dtype=float))

    def normalize(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.minimum) / (self.maximum - self.minimum)

    def inverse(self, X: np.ndarray) -> np.ndarray:
        return self.minimum + np.asarray(X, dtype=float) * (self.maximum - self.minimum)


@dataclass(frozen=True)
class SimulationResult:
    values: np.ndarray
    trajectories: tuple[np.ndarray, ...]
    success: bool
    solver: str
    message: str = ""


def arrhenius(log10_a: float, log10_ea: float, temperature: float) -> float:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    value = 10.0**log10_a * np.exp(-(10.0**log10_ea) / (8.314462618 * temperature))
    if not np.isfinite(value):
        raise FloatingPointError("non-finite Arrhenius rate")
    return float(value)


def integrate_ode(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    y0: Sequence[float],
    time: float,
    *,
    method: str = "LSODA",
) -> tuple[np.ndarray, str]:
    y0 = np.asarray(y0, dtype=float)
    if time < 0 or not np.all(np.isfinite(y0)) or np.any(y0 < 0):
        raise ValueError("time and initial concentrations must be finite and non-negative")
    if time == 0:
        return y0.copy(), "initial"
    if solve_ivp is not None:
        sol = solve_ivp(rhs, (0.0, float(time)), y0, t_eval=[float(time)], method=method, rtol=1e-7, atol=1e-10)
        if sol.success and np.all(np.isfinite(sol.y[:, -1])):
            return np.maximum(sol.y[:, -1], 0.0), "scipy"
    state = y0.copy()
    n_steps = max(16, min(2048, int(np.ceil(time / max(time / 256.0, 1e-3)))))
    dt = float(time) / n_steps
    for i in range(n_steps):
        t = i * dt
        k1 = np.asarray(rhs(t, state), dtype=float)
        k2 = np.asarray(rhs(t + dt / 2, np.maximum(state + dt * k1 / 2, 0)), dtype=float)
        k3 = np.asarray(rhs(t + dt / 2, np.maximum(state + dt * k2 / 2, 0)), dtype=float)
        k4 = np.asarray(rhs(t + dt, np.maximum(state + dt * k3, 0)), dtype=float)
        state = np.maximum(state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6, 0)
        if not np.all(np.isfinite(state)):
            raise RuntimeError("ODE integration produced non-finite concentrations")
    return state, "rk4"


class KineticModel(ABC):
    input_names: tuple[str, ...] = ()
    species_names: tuple[str, ...] = ()

    def __init__(self, parameters: Sequence[KineticParameter], input_transform: InputTransform | None = None) -> None:
        self.parameters = tuple(parameters)
        self.param_names = [p.name for p in self.parameters]
        self._W = np.asarray([p.default for p in self.parameters], dtype=float)
        self.input_transform = input_transform

    @property
    def W(self) -> np.ndarray:
        return self._W

    @W.setter
    def W(self, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=float)
        self.validate_parameters(value)
        self._W = value.copy()

    @property
    def parameter_bounds(self) -> np.ndarray:
        return np.asarray([p.bounds for p in self.parameters], dtype=float)

    def validate_parameters(self, W: np.ndarray) -> None:
        W = np.asarray(W, dtype=float)
        if W.shape != (len(self.parameters),) or not np.all(np.isfinite(W)):
            raise ValueError(f"W must be a finite vector of length {len(self.parameters)}")
        bounds = self.parameter_bounds
        if np.any(W < bounds[:, 0]) or np.any(W > bounds[:, 1]):
            raise ValueError("kinetic parameters are outside declared bounds")

    def validate_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.input_names) or not np.all(np.isfinite(X)):
            raise ValueError(f"X must be finite with shape (n, {len(self.input_names)})")
        return self.input_transform.inverse(X) if self.input_transform is not None else X

    def compute_mean(self, X: np.ndarray, W: np.ndarray) -> np.ndarray:
        return self.simulate(X, W).values

    @abstractmethod
    def simulate(self, X: np.ndarray, W: np.ndarray | None = None) -> SimulationResult:
        raise NotImplementedError

    def compute_gradients_W(self, X: np.ndarray, W: np.ndarray) -> np.ndarray:
        W = np.asarray(W, dtype=float)
        self.validate_parameters(W)
        gradients = np.empty((len(X), len(W)), dtype=float)
        for i in range(len(W)):
            span = self.parameters[i].bounds[1] - self.parameters[i].bounds[0]
            h = max(1e-6, 1e-4 * max(abs(W[i]), 1.0))
            if np.isfinite(span):
                h = min(h, span / 1000)
            wp, wm = W.copy(), W.copy()
            wp[i] = min(W[i] + h, self.parameters[i].bounds[1])
            wm[i] = max(W[i] - h, self.parameters[i].bounds[0])
            gradients[:, i] = ((self.compute_mean(X, wp) - self.compute_mean(X, wm)) / (wp[i] - wm[i])).ravel()
        return gradients
