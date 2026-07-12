"""Minimal CIGP runs for representative kinetic priors."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pc_mcmc_cigp.cigp import CIGPConfig, CIGPRegressor
from pc_mcmc_cigp.kinetics import TemplateRegistry, create_kinetic_template


CASES = {
    "simple_arrhenius": [1.0, 1.0, 350.0],
    "reversible_arrhenius": [1.0, 1.0, 350.0],
    "michaelis_menten": [1.0, 0.1, 350.0],
    "langmuir_hinshelwood": [1.0, 1.0, 350.0],
    "epoxidation": [1.0, 1.0, 350.0],
}


def main() -> None:
    for name, conditions in CASES.items():
        physics = create_kinetic_template(name)
        X = np.asarray([conditions + [time] for time in (0.01, 0.02, 0.04, 0.08)])
        y = physics.compute_mean(X, physics.W).ravel()
        model = CIGPRegressor(physics, CIGPConfig(optimize_hyperparameters=False)).fit(X, y)
        mean, variance = model.predict(X[-1:])
        print(f"{name:26s} prediction={mean.item():.6g} variance={variance.item():.3g}")

    print("\nAvailable templates:")
    for name in TemplateRegistry.names():
        print(TemplateRegistry.describe(name))


if __name__ == "__main__":
    main()
