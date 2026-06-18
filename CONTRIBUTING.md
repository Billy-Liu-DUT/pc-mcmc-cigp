# Contributing

This repository is being converted from a research prototype into a reusable
open-source package. Contributions should keep the package API small,
reproducible, and testable.

## Development Setup

```bash
pip install -e ".[dev,benchmarks]"
pytest
```

If `pytest` is not installed in the local conda environment, the current tests
can also be executed as plain Python functions, but `pytest` is recommended for
normal development.

## Before Opening a Pull Request

- Add or update tests for changed behavior.
- Keep long MCMC or BO runs in `experiments/`, not in unit tests.
- Do not commit generated outputs, figures, checkpoints, or `__pycache__`.
- Keep public APIs explicit and documented in `README.md` or `docs/USAGE_ZH.md`.

## Experiment Outputs

Generated files should go under:

```text
examples/outputs/
```

Only `examples/outputs/README.md` is tracked.
