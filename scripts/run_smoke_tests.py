from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    tests_dir = root / "tests"
    total = 0
    for path in sorted(tests_dir.glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        for name in sorted(item for item in dir(module) if item.startswith("test_")):
            getattr(module, name)()
            total += 1
            print(f"PASS {path.name}::{name}")
    print(f"\n{total} smoke tests passed.")


if __name__ == "__main__":
    main()
