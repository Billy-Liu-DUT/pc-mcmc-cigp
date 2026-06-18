# PC-MCMC-CIGP

![CI](https://github.com/Billy-Liu-DUT/pc-mcmc-cigp/actions/workflows/ci.yml/badge.svg)

<p align="center">
  <img src="assets/logo.svg" alt="PC-MCMC-CIGP logo" width="128">
</p>

<p align="center">
  <img src="assets/workflow.svg" alt="PC-MCMC-CIGP workflow">
</p>

PC-MCMC-CIGP is a research-oriented Python package for interpretable reaction
network discovery and physics-informed active learning.

The project implements two core algorithmic modules from the paper draft
`Synergizing Physically Constrained MCMC and Chemical-Informed Gaussian Processes
for Reaction Network Discovery`:

1. **PC-MCMC**: spike-and-slab Bayesian topology search over candidate elementary
   reactions, with mass/charge conservation and detailed-balance style physical
   constraints.
2. **CIGP**: Chemical-Informed Gaussian Processes, where an ODE or kinetic
   physics model is embedded as the GP prior mean and a kernel models systematic
   residual error.

SINDy, PySR, and standard GP baselines are kept as examples or optional
benchmarks rather than core APIs.

## Installation

```bash
pip install -e .
```

For benchmark baselines:

```bash
pip install -e ".[benchmarks]"
```

For development:

```bash
pip install -e ".[dev,benchmarks]"
pytest
```

If your environment does not have `pytest`, run the built-in smoke runner:

```bash
python scripts/run_smoke_tests.py
```

## Quick Start

```python
import numpy as np

from pc_mcmc_cigp.discovery import MCMCConfig, MechanismEngine, SpikeAndSlabSampler
from pc_mcmc_cigp.reactions import Reaction, Species

a = Species("A", {"A": 1})
b = Species("B", {"B": 1})
engine = MechanismEngine([a, b], [Reaction([a], [b])])

t = np.linspace(0, 1, 8)
y0 = np.array([1.0, 0.0])
data = engine.simulate(np.array([1.0]), np.array([1.0]), y0, t)
dataset = [{"t": t, "y0_full": y0, "data_matrix": data, "obs_indices": [0, 1]}]

result = SpikeAndSlabSampler(
    engine,
    MCMCConfig(n_steps=200, burn_in=50, enable_thermo_constraints=False, random_state=0),
).fit(dataset)

print(result.posterior_inclusion_probabilities)
print([r.equation_str for r in result.selected_reactions])
```

## Package Layout

- `pc_mcmc_cigp.reactions`: species, reaction, and atom-balanced network generation.
- `pc_mcmc_cigp.discovery`: mass-action ODE engine and spike-and-slab sampler.
- `pc_mcmc_cigp.cigp`: sklearn-style CIGP regressor.
- `pc_mcmc_cigp.acquisition`: EI, GWU, discrepancy hunter, and physically constrained EI.
- `pc_mcmc_cigp.benchmarks`: HBr mechanism discovery and styrene epoxidation examples.
- `examples/`: runnable scripts for paper-style experiments and baselines.

Chinese documentation:

- `docs/USAGE_ZH.md`
- `docs/API_ZH.md`
- `docs/PAPER_JOURNAL_STRATEGY_ZH.md`

## Reproducing the Main Examples

```bash
python examples/discover_hbr_mechanism.py
python examples/optimize_epoxidation_cigp.py
python examples/baselines/sindy_hbr.py
```

The first two scripts use the package APIs directly. The SINDy script is a
baseline and requires the optional `pysindy` dependency.

For paper-style machine-readable outputs:

```bash
python experiments/fig3_hbr_discovery.py
python experiments/fig4_epoxidation_bo.py
python scripts/plot_fig3.py
python scripts/plot_fig4.py
```

These write JSON/CSV outputs under `examples/outputs/`. Generated outputs are
ignored by git; keep the scripts as the source of truth.

Original exploratory scripts are preserved in `legacy/` for reference. New code
should use the package API under `pc_mcmc_cigp/`.

To regenerate the Fig.4 comparison from the source PowerPoint data:

```bash
python scripts/extract_ppt_optimization_logs.py "D:/工作二连续流/图/图改.pptx" data
python scripts/plot_fig4.py
```

## Development Status

This is the first open-source-oriented cleanup of a research codebase. The ODE
engine uses SciPy's `solve_ivp` when available and falls back to a lightweight
RK4 integrator for minimal environments. Longer MCMC runs, publication-quality
figures, and full paper reproduction should be treated as experiment scripts
rather than CI tests.
