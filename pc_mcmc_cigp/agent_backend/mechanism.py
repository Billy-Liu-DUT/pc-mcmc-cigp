from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from pc_mcmc_cigp.agent_backend.models import CompilationReport, MechanismSpec
from pc_mcmc_cigp.kinetics import ArrheniusRate, MassActionRate, PowerLawRate, ReversibleRate, SaturationRate
from pc_mcmc_cigp.reactions import PathwayGenerator, Reaction, Species
from pc_mcmc_cigp.discovery import MechanismEngine


@dataclass(frozen=True)
class CompiledMechanism:
    engine: MechanismEngine
    report: CompilationReport
    step_ids: tuple[str, ...]


class MechanismCompiler:
    SUPPORTED_RATE_LAWS = {"mass_action", "power_law", "arrhenius", "reversible", "saturation", "inhibition"}

    def compile(self, spec: MechanismSpec) -> CompiledMechanism:
        errors, warnings = [], []
        names = [item.name for item in spec.species]
        if len(names) != len(set(names)):
            errors.append("species names must be unique")
        species = [Species(item.name, {**item.formula, "Charge": item.charge}) for item in spec.species]
        by_name = {item.name: item for item in species}; reactions = []
        for step in spec.steps:
            unknown = (set(step.reactants) | set(step.products) | set(step.catalysts)) - set(by_name)
            if unknown: errors.append(f"{step.step_id}: unknown species {sorted(unknown)}"); continue
            if step.rate_law_family not in self.SUPPORTED_RATE_LAWS:
                errors.append(f"{step.step_id}: unsupported rate law {step.rate_law_family}"); continue
            try:
                reactants = self._expand(step.reactants, by_name) + self._expand(step.catalysts, by_name)
                products = self._expand(step.products, by_name) + self._expand(step.catalysts, by_name)
            except ValueError as exc:
                errors.append(f"{step.step_id}: {exc}"); continue
            reaction = Reaction(reactants, products)
            atom_types = sorted({atom for item in species for atom in item.atoms})
            if atom_types and not reaction.is_balanced(atom_types):
                errors.append(f"{step.step_id}: atom or charge balance failed")
                continue
            rate_law = self._make_rate_law(step.rate_law_family, reactants, products, by_name, species)
            reactions.append(Reaction(reactants, products, rate_law=rate_law))
            if step.prior_probability <= 0 or step.prior_probability >= 1:
                warnings.append(f"{step.step_id}: prior_probability should lie strictly between 0 and 1")
        if not reactions: errors.append("mechanism has no compilable reaction steps")
        if errors:
            report = CompilationReport(False, tuple(errors), tuple(warnings), tuple(names), [], ())
            return CompiledMechanism(MechanismEngine(species, []), report, ())
        engine = MechanismEngine(species, reactions)
        report = CompilationReport(True, (), tuple(warnings), tuple(names), engine.S.tolist(), engine.rate_parameter_names)
        return CompiledMechanism(engine, report, tuple(step.step_id for step in spec.steps))

    def candidate_pathways(self, compiled: CompiledMechanism, sources: Sequence[str], targets: Sequence[str], max_steps: int = 6) -> list[tuple[int, ...]]:
        paths = PathwayGenerator(compiled.engine.species, compiled.engine.reactions).generate(sources, targets, max_steps=max_steps)
        return [path.reaction_indices for path in paths]

    @staticmethod
    def _expand(stoichiometry: dict[str, float], by_name: dict[str, Species]) -> list[Species]:
        output = []
        for name, coefficient in stoichiometry.items():
            if coefficient <= 0 or not float(coefficient).is_integer():
                raise ValueError("current compiler requires positive integer stoichiometric coefficients")
            output.extend([by_name[name]] * int(coefficient))
        return output

    @staticmethod
    def _orders(items: Sequence[Species], species: Sequence[Species]) -> dict[int, float]:
        counts = Counter(item.name for item in items); indices = {item.name: i for i, item in enumerate(species)}
        return {indices[name]: float(count) for name, count in counts.items()}

    def _make_rate_law(self, family, reactants, products, by_name, species):
        forward_orders = self._orders(reactants, species)
        if family == "mass_action": return MassActionRate(forward_orders)
        if family == "arrhenius": return ArrheniusRate(forward_orders)
        if family == "power_law": return PowerLawRate(tuple(forward_orders))
        if family == "reversible": return ReversibleRate(MassActionRate(forward_orders), MassActionRate(self._orders(products, species)))
        substrate_index = next(iter(forward_orders))
        inhibitors = tuple(index for index in forward_orders if index != substrate_index) if family == "inhibition" else ()
        return SaturationRate(substrate_index, inhibitors)


def build_mcmc_dataset(compiled: CompiledMechanism, rows: Sequence[dict[str, float | str]]) -> list[dict]:
    if not compiled.report.valid:
        raise ValueError("cannot build a dataset for an invalid mechanism")
    groups: dict[tuple[str, float], list[dict[str, float | str]]] = {}
    for row in rows:
        key = (str(row["experiment_id"]), float(row.get("replicate", 1)))
        groups.setdefault(key, []).append(row)
    dataset = []
    for _, group in groups.items():
        group = sorted(group, key=lambda item: float(item["time_s"])); t = np.asarray([row["time_s"] for row in group], dtype=float)
        observed_indices, observed_rows = [], []
        for index, name in enumerate(compiled.report.species_order):
            column = f"{name}_mol_L"
            if all(column in row for row in group):
                observed_indices.append(index); observed_rows.append([float(row[column]) for row in group])
        if not observed_indices: raise ValueError("dataset has no concentration columns matching mechanism species")
        y0 = np.zeros(len(compiled.report.species_order))
        for index, name in enumerate(compiled.report.species_order):
            concentration = f"{name}_mol_L"; initial = f"{name}0_mol_L"
            if concentration in group[0] and np.isclose(t[0], 0): y0[index] = float(group[0][concentration])
            elif initial in group[0]: y0[index] = float(group[0][initial])
        experiment = {"t": t, "y0_full": y0, "data_matrix": np.asarray(observed_rows), "obs_indices": observed_indices}
        if "temperature_K" in group[0]: experiment["temperature"] = float(group[0]["temperature_K"])
        dataset.append(experiment)
    return dataset
