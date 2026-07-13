from __future__ import annotations


class SchemaValidationError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


def validate_schema(value, schema: dict, path: str = "$") -> list[str]:
    errors: list[str] = []
    expected = schema.get("type")
    type_map = {
        "object": dict,
        "array": list,
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "null": type(None),
    }
    if isinstance(expected, list):
        if not any(_matches_type(value, item, type_map) for item in expected):
            return [f"{path}: expected one of {expected}"]
    elif expected and not _matches_type(value, expected, type_map):
        return [f"{path}: expected {expected}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value must be one of {schema['enum']}")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required key {key}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    errors.append(f"{path}: unexpected key {key}")
        for key, child_schema in properties.items():
            if key in value:
                errors.extend(validate_schema(value[key], child_schema, f"{path}.{key}"))
    elif isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            errors.extend(validate_schema(item, schema["items"], f"{path}[{index}]"))
    return errors


def _matches_type(value, expected: str, type_map: dict) -> bool:
    target = type_map.get(expected)
    if target is None:
        return True
    if expected in {"number", "integer"} and isinstance(value, bool):
        return False
    return isinstance(value, target)
