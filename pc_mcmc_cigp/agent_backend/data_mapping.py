from __future__ import annotations

import csv
import re
from pathlib import Path

from pc_mcmc_cigp.agent_backend.models import ColumnMapping, DataMappingReport, ReactionProjectSpec


class BenchmarkDataMapper:
    """Map user CSV files to the HBr time-series and epoxidation design contracts."""

    EXACT_ALIASES = {
        "experiment": ("experiment_id", "identity"),
        "experimentid": ("experiment_id", "identity"),
        "exp_id": ("experiment_id", "identity"),
        "run_id": ("experiment_id", "identity"),
        "time": ("time_s", "identity"),
        "time_s": ("time_s", "identity"),
        "time_sec": ("time_s", "identity"),
        "time_min": ("time_s", "minutes_to_seconds"),
        "temperature": ("temperature_K", "identity"),
        "temperature_k": ("temperature_K", "identity"),
        "temp_k": ("temperature_K", "identity"),
        "temperature_c": ("temperature_K", "celsius_to_kelvin"),
        "temp_c": ("temperature_K", "celsius_to_kelvin"),
        "replicate": ("replicate", "identity"),
        "repeat": ("replicate", "identity"),
        "yield": ("yield", "identity"),
        "yield_percent": ("yield", "percent_to_fraction"),
    }

    def map_csv(self, project: ReactionProjectSpec, path: str | Path, overrides: dict | None = None) -> DataMappingReport:
        path = Path(path)
        if not path.exists():
            return DataMappingReport(False, (), (), ("dataset file does not exist",), (), ())
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            raw_rows = list(reader)
            columns = tuple(reader.fieldnames or ())
        if not raw_rows:
            return DataMappingReport(False, (), (), ("dataset contains no rows",), (), ())
        overrides = overrides or {}
        mappings = tuple(
            self._override_mapping(column, overrides[column])
            if column in overrides
            else self._map_column(project, column)
            for column in columns
        )
        errors, warnings = [], []
        targets = {item.target for item in mappings}
        if "experiment_id" not in targets:
            errors.append("cannot identify experiment_id column")
        if "time_s" not in targets:
            errors.append("cannot identify time column and unit")
        if any(item.requires_confirmation for item in mappings):
            warnings.append("ambiguous column mappings require user confirmation")
        unresolved = tuple(item.source for item in mappings if item.target.startswith("unresolved:"))
        normalized = []
        for row_index, row in enumerate(raw_rows, start=2):
            converted = {}
            for mapping in mappings:
                raw = row.get(mapping.source, "")
                if raw in (None, ""):
                    continue
                try:
                    converted[mapping.target] = self._convert(raw, mapping.conversion, mapping.target)
                except ValueError:
                    errors.append(
                        f"row {row_index}: {mapping.source} cannot be converted using {mapping.conversion}"
                    )
            normalized.append(converted)
        valid = not errors and not any(item.requires_confirmation for item in mappings)
        return DataMappingReport(
            valid, mappings, unresolved, tuple(dict.fromkeys(errors)), tuple(warnings), tuple(normalized)
        )

    @staticmethod
    def _override_mapping(source: str, override) -> ColumnMapping:
        if isinstance(override, str):
            return ColumnMapping(source, override, "identity", 1.0, False)
        conversion = override.get("conversion", "identity")
        allowed = {"identity", "minutes_to_seconds", "celsius_to_kelvin", "percent_to_fraction", "numeric_or_text"}
        if conversion not in allowed:
            raise ValueError(f"unsupported mapping conversion {conversion!r}")
        return ColumnMapping(source, override["target"], conversion, 1.0, False)

    def _map_column(self, project: ReactionProjectSpec, source: str) -> ColumnMapping:
        token = self._token(source)
        if token in self.EXACT_ALIASES:
            target, conversion = self.EXACT_ALIASES[token]
            ambiguous = token in {"time", "temperature"}
            return ColumnMapping(source, target, conversion, 0.7 if ambiguous else 1.0, ambiguous)
        species = [*project.reactants, *project.known_products, *project.suspected_intermediates]
        for item in species:
            species_token = self._token(item.name)
            if token in {species_token, f"{species_token}_mol_l", f"conc_{species_token}"}:
                return ColumnMapping(source, f"{item.name}_mol_L")
            if token in {f"{species_token}0", f"{species_token}0_mol_l", f"initial_{species_token}"}:
                return ColumnMapping(source, f"{item.name}0_mol_L")
        # Preserve numeric experimental conditions for CIGP rather than discarding them.
        if re.fullmatch(r"[a-z][a-z0-9_]*", token):
            return ColumnMapping(source, token, "numeric_or_text", 0.8, False)
        return ColumnMapping(source, f"unresolved:{source}", "identity", 0.0, True)

    @staticmethod
    def _token(value: str) -> str:
        return re.sub(r"[^a-z0-9_]+", "_", value.strip().lower()).strip("_")

    @staticmethod
    def _convert(raw: str, conversion: str, target: str):
        if target == "experiment_id" or conversion == "identity" and target.startswith("unresolved:"):
            return str(raw).strip()
        if conversion == "numeric_or_text":
            try:
                return float(raw)
            except ValueError:
                return str(raw).strip()
        value = float(raw)
        if conversion == "minutes_to_seconds":
            return value * 60.0
        if conversion == "celsius_to_kelvin":
            return value + 273.15
        if conversion == "percent_to_fraction":
            return value / 100.0
        return value


def build_cigp_training_data(rows, model: object, target_column: str):
    """Compile normalized benchmark rows into the ordered X/y contract used by CIGP."""
    import numpy as np

    rows = list(rows)
    groups = {}
    for row in rows:
        groups.setdefault((str(row["experiment_id"]), float(row.get("replicate", 1))), []).append(row)
    X, y = [], []
    for group in groups.values():
        ordered = sorted(group, key=lambda item: float(item["time_s"]))
        first = ordered[0]
        for row in ordered:
            if target_column not in row:
                continue
            features = []
            for input_name in model.input_names:
                if input_name == "time":
                    features.append(float(row["time_s"]))
                elif input_name == "temperature":
                    features.append(float(row["temperature_K"]))
                elif input_name.endswith("0"):
                    species = input_name[:-1]
                    initial_key = f"{species}0_mol_L"
                    observed_key = f"{species}_mol_L"
                    if initial_key in first:
                        features.append(float(first[initial_key]))
                    elif observed_key in first and np.isclose(float(first["time_s"]), 0):
                        features.append(float(first[observed_key]))
                    else:
                        features.append(0.0)
                elif input_name in row:
                    features.append(float(row[input_name]))
                else:
                    raise ValueError(f"dataset is missing CIGP input {input_name}")
            X.append(features)
            y.append(float(row[target_column]))
    if not X:
        raise ValueError(f"dataset contains no observations for target {target_column}")
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)
