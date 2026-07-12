from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from pc_mcmc_cigp.kinetics.core import arrhenius


class RateLaw:
    parameter_names: tuple[str, ...] = ("k",)
    parameter_bounds: tuple[tuple[float, float], ...] = ((1e-12, 1e6),)

    def rate(self, concentrations: np.ndarray, parameters: Sequence[float], context: Mapping[str, float] | None = None) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class MassActionRate(RateLaw):
    orders: Mapping[int, float]

    def rate(self, concentrations, parameters, context=None) -> float:
        value = float(parameters[0])
        for idx, order in self.orders.items():
            value *= max(float(concentrations[idx]), 0.0) ** order
        return value


@dataclass(frozen=True)
class PowerLawRate(RateLaw):
    species_indices: tuple[int, ...]
    fixed_orders: tuple[float, ...] | None = None

    @property
    def parameter_names(self):
        return ("k",) if self.fixed_orders is not None else ("k",) + tuple(f"order_{i}" for i in self.species_indices)

    @property
    def parameter_bounds(self):
        return ((1e-12, 1e6),) if self.fixed_orders is not None else ((1e-12, 1e6),) + ((0.0, 4.0),) * len(self.species_indices)

    def rate(self, concentrations, parameters, context=None) -> float:
        orders = self.fixed_orders if self.fixed_orders is not None else parameters[1:]
        value = float(parameters[0])
        for idx, order in zip(self.species_indices, orders):
            value *= max(float(concentrations[idx]), 0.0) ** float(order)
        return value


@dataclass(frozen=True)
class ArrheniusRate(RateLaw):
    orders: Mapping[int, float]
    parameter_names: tuple[str, ...] = ("log10_A", "log10_Ea")
    parameter_bounds: tuple[tuple[float, float], ...] = ((-6.0, 18.0), (0.0, 6.0))

    def rate(self, concentrations, parameters, context=None) -> float:
        if context is None or "temperature" not in context:
            raise ValueError("ArrheniusRate requires context['temperature']")
        value = arrhenius(parameters[0], parameters[1], context["temperature"])
        for idx, order in self.orders.items():
            value *= max(float(concentrations[idx]), 0.0) ** order
        return value


@dataclass(frozen=True)
class ReversibleRate(RateLaw):
    forward: RateLaw
    reverse: RateLaw

    @property
    def parameter_names(self):
        return tuple(f"f_{x}" for x in self.forward.parameter_names) + tuple(f"r_{x}" for x in self.reverse.parameter_names)

    @property
    def parameter_bounds(self):
        return tuple(self.forward.parameter_bounds) + tuple(self.reverse.parameter_bounds)

    def rate(self, concentrations, parameters, context=None) -> float:
        n = len(self.forward.parameter_names)
        return self.forward.rate(concentrations, parameters[:n], context) - self.reverse.rate(concentrations, parameters[n:], context)


@dataclass(frozen=True)
class SaturationRate(RateLaw):
    substrate_index: int
    inhibitor_indices: tuple[int, ...] = field(default_factory=tuple)
    parameter_names: tuple[str, ...] = ("vmax", "Km", "Ki")
    parameter_bounds: tuple[tuple[float, float], ...] = ((1e-12, 1e6), (1e-12, 1e6), (1e-12, 1e6))

    def rate(self, concentrations, parameters, context=None) -> float:
        s = max(float(concentrations[self.substrate_index]), 0.0)
        inhibition = sum(max(float(concentrations[i]), 0.0) for i in self.inhibitor_indices)
        return float(parameters[0]) * s / (float(parameters[1]) * (1 + inhibition / float(parameters[2])) + s)
