from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np

from pc_mcmc_cigp.agent_backend.models import DataValidationReport, ExperimentRequest, ReactionProjectSpec


class ExperimentPlanner:
    """Create a conservative initial kinetic experiment matrix from a project spec."""

    def plan(self, project: ReactionProjectSpec, *, temperatures: Sequence[float] | None = None) -> list[ExperimentRequest]:
        temperatures = tuple(temperatures or self._temperature_levels(project))
        times = tuple(float(x) for x in project.known_conditions.get("sampling_times", [0, 30, 60, 120, 300, 600]))
        reactants = [item.name for item in project.reactants]
        required = tuple(dict.fromkeys([*reactants, *(item.name for item in project.known_products)]))
        optional = tuple(item.name for item in project.suspected_intermediates)
        requests = []
        for idx, temperature in enumerate(temperatures, start=1):
            variables = {"temperature_K": temperature}
            for reactant in reactants:
                variables[f"{reactant}0_mol_L"] = 1.0
            for catalyst in project.catalysts:
                variables[f"{catalyst.name}0_mol_L"] = 0.01
            requests.append(ExperimentRequest(
                experiment_id=f"E{idx:03d}", purpose="temperature-dependent kinetic profile",
                variables=variables, sampling_times=times, required_observables=required,
                optional_observables=optional, replicates=2,
                rationale="Multiple temperatures and time points separate kinetic parameters from endpoint yield.",
            ))
        if len(reactants) >= 2:
            base = len(requests)
            for offset, factor in enumerate((0.5, 1.5), start=1):
                variables = {"temperature_K": temperatures[len(temperatures) // 2]}
                for i, reactant in enumerate(reactants):
                    variables[f"{reactant}0_mol_L"] = factor if i == 0 else 1.0
                requests.append(ExperimentRequest(
                    experiment_id=f"E{base + offset:03d}", purpose="reaction-order discrimination",
                    variables=variables, sampling_times=times, required_observables=required,
                    optional_observables=optional, replicates=2,
                    rationale=f"Perturb {reactants[0]} independently to distinguish kinetic orders.",
                ))
        return requests

    @staticmethod
    def _temperature_levels(project: ReactionProjectSpec) -> tuple[float, ...]:
        temp_vars = [v for v in project.controllable_variables if v.name.lower() in {"temperature", "temperature_k"}]
        if temp_vars and temp_vars[0].lower is not None and temp_vars[0].upper is not None:
            lo, hi = temp_vars[0].lower, temp_vars[0].upper
            return (float(lo), float((lo + hi) / 2), float(hi))
        return (298.15, 323.15, 348.15)


def write_experiment_template(project: ReactionProjectSpec, requests: Sequence[ExperimentRequest], path: str | Path) -> Path:
    path = Path(path)
    observable_columns = [f"{obs.species}_mol_L" for obs in project.observed_variables]
    if not observable_columns:
        observable_columns = [f"{item.name}_mol_L" for item in [*project.reactants, *project.known_products, *project.suspected_intermediates]]
    variable_columns = sorted({key for request in requests for key in request.variables})
    fields = ["experiment_id", "replicate", "time_s", *variable_columns, *dict.fromkeys(observable_columns)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for request in requests:
            for replicate in range(1, request.replicates + 1):
                for time in request.sampling_times:
                    row = {"experiment_id": request.experiment_id, "replicate": replicate, "time_s": time, **request.variables}
                    writer.writerow(row)
    return path


class ExperimentDataValidator:
    def validate_rows(self, project: ReactionProjectSpec, rows: Sequence[dict]) -> tuple[DataValidationReport, list[dict]]:
        errors, warnings, risks = [], [], []
        columns = {key for row in rows for key in row}
        missing = {"experiment_id", "time_s"} - columns
        if missing: errors.append(f"missing required columns: {', '.join(sorted(missing))}")
        expected_species = [item.name for item in [*project.reactants, *project.known_products]]
        species_columns = {name: self._find_species_column(columns, name) for name in expected_species}
        missing_species = [name for name, column in species_columns.items() if column is None]
        if missing_species and project.workflow_mode.value != "optimization_only":
            errors.append(f"missing concentration columns: {', '.join(missing_species)}")
        groups: dict[str, list[dict]] = {}
        for index, row in enumerate(rows, start=2):
            try:
                if float(row.get("time_s", 0.0)) < 0: errors.append(f"row {index}: negative time")
            except (TypeError, ValueError): errors.append(f"row {index}: time_s is not numeric")
            for name, column in species_columns.items():
                if column and column in row and float(row[column]) < 0: errors.append(f"row {index}: negative {name} concentration")
            groups.setdefault(str(row.get("experiment_id", "")), []).append(row)
        for experiment_id, group in groups.items():
            times = [float(row.get("time_s", np.nan)) for row in group]
            if not any(np.isclose(times, 0.0)): warnings.append(f"{experiment_id}: no zero-time observation")
            if times != sorted(times): warnings.append(f"{experiment_id}: rows are not ordered by time")
        coverage = {
            name: 0.0 if column is None or not rows else sum(column in row for row in rows) / len(rows)
            for name, column in species_columns.items()
        }
        temperatures = {row.get("temperature_K") for row in rows if "temperature_K" in row}
        if len(temperatures) < 3: risks.append("Arrhenius A and Ea may not be identifiable with fewer than three temperatures")
        if len(groups) < 3: risks.append("Mechanism discrimination is weak with fewer than three experimental conditions")
        observed_candidates = any(self._find_species_column(columns, item.name) for item in project.suspected_intermediates)
        if project.suspected_intermediates and not observed_candidates: risks.append("Suspected intermediates are unobserved")
        if project.workflow_mode.value != "optimization_only" and all(len(group) <= 1 for group in groups.values()):
            risks.append("Endpoint-only data are insufficient for reliable mechanism discrimination")
        if not rows: errors.append("dataset contains no rows")
        valid = not errors and bool(rows)
        report = DataValidationReport(valid, tuple(dict.fromkeys(errors)), tuple(dict.fromkeys(warnings)), None, coverage, {}, tuple(risks), None)
        return report, list(rows)

    def validate(self, project: ReactionProjectSpec, path: str | Path) -> tuple[DataValidationReport, list[dict[str, float | str]]]:
        path = Path(path); errors, warnings, risks = [], [], []
        if not path.exists():
            return DataValidationReport(False, ("dataset file does not exist",), (), None, {}, {}, (), None), []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle)); columns = set(handle.seek(0) or csv.DictReader(handle).fieldnames or [])
        required_meta = {"experiment_id", "time_s"}
        missing = required_meta - columns
        if missing: errors.append(f"missing required columns: {', '.join(sorted(missing))}")
        expected_species = [item.name for item in [*project.reactants, *project.known_products]]
        species_columns = {name: self._find_species_column(columns, name) for name in expected_species}
        missing_species = [name for name, column in species_columns.items() if column is None]
        if missing_species: errors.append(f"missing concentration columns: {', '.join(missing_species)}")
        normalized: list[dict[str, float | str]] = []
        groups: dict[str, list[dict[str, float | str]]] = {}
        for index, row in enumerate(rows, start=2):
            converted: dict[str, float | str] = {"experiment_id": row.get("experiment_id", "")}
            for key, value in row.items():
                if key == "experiment_id" or value in (None, ""):
                    continue
                try: converted[key] = float(value)
                except ValueError: errors.append(f"row {index}: {key} is not numeric")
            if float(converted.get("time_s", 0.0)) < 0: errors.append(f"row {index}: negative time")
            for name, column in species_columns.items():
                if column and column in converted and float(converted[column]) < 0: errors.append(f"row {index}: negative {name} concentration")
            normalized.append(converted); groups.setdefault(str(converted["experiment_id"]), []).append(converted)
        for experiment_id, group in groups.items():
            times = [float(row.get("time_s", np.nan)) for row in group]
            if not any(np.isclose(times, 0.0)): warnings.append(f"{experiment_id}: no zero-time observation")
            if times != sorted(times): warnings.append(f"{experiment_id}: rows are not ordered by time")
        coverage = {}
        for name, column in species_columns.items():
            coverage[name] = 0.0 if column is None or not rows else sum(bool(row.get(column, "")) for row in rows) / len(rows)
        temperatures = {row.get("temperature_K") for row in normalized if "temperature_K" in row}
        if len(temperatures) < 3: risks.append("Arrhenius A and Ea may not be identifiable with fewer than three temperatures")
        if len(groups) < 3: risks.append("Mechanism discrimination is weak with fewer than three experimental conditions")
        if not any(item.name in species_columns and species_columns[item.name] for item in project.suspected_intermediates):
            if project.suspected_intermediates: risks.append("Suspected intermediates are unobserved")
        valid = not errors and bool(rows)
        if not rows: errors.append("dataset contains no rows"); valid = False
        report = DataValidationReport(valid, tuple(dict.fromkeys(errors)), tuple(dict.fromkeys(warnings)), str(path) if valid else None, coverage, {}, tuple(risks), None)
        return report, normalized

    @staticmethod
    def _find_species_column(columns: set[str], species: str) -> str | None:
        candidates = (f"{species}_mol_L", species)
        return next((item for item in candidates if item in columns), None)
