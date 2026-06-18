from __future__ import annotations

import numpy as np

from pc_mcmc_cigp.discovery import MCMCConfig, MechanismEngine, SpikeAndSlabSampler
from pc_mcmc_cigp.reactions import Reaction, Species


class HBrMechanismBenchmark:
    """Small H2/Br2 radical-chain mechanism discovery benchmark."""

    def __init__(self, random_state: int | None = None, noise_level: float = 0.0) -> None:
        self.rng = np.random.default_rng(random_state)
        self.noise_level = noise_level
        self.species = [
            Species("H2", {"H": 2}),
            Species("Br2", {"Br": 2}),
            Species("HBr", {"H": 1, "Br": 1}),
            Species("H.", {"H": 1}),
            Species("Br.", {"Br": 1}),
        ]
        h2, br2, hbr, h_rad, br_rad = self.species
        self.reactions = [
            Reaction([br2], [br_rad, br_rad]),
            Reaction([br_rad, br_rad], [br2]),
            Reaction([br_rad, h2], [hbr, h_rad]),
            Reaction([h_rad, br2], [hbr, br_rad]),
            Reaction([h2, br2], [hbr, hbr]),
        ]
        self.engine = MechanismEngine(self.species, self.reactions, stiffness_cap_k=1e3)

    def make_dataset(self, n_points: int = 8) -> list[dict]:
        t = np.linspace(0.0, 0.2, n_points)
        y0 = np.zeros(len(self.species))
        y0[self.engine.s_map["H2"]] = 1.0
        y0[self.engine.s_map["Br2"]] = 1.0
        z = np.array([1, 1, 1, 1, 0], dtype=float)
        k = np.array([10.0, 100.0, 1.0, 50.0, 0.1])
        data = self.engine.simulate(k, z, y0, t)
        if self.noise_level:
            data = np.maximum(data + self.rng.normal(0.0, self.noise_level, size=data.shape), 0.0)
        obs_indices = [self.engine.s_map[name] for name in ["H2", "Br2", "HBr"]]
        return [{"t": t, "y0_full": y0, "data_matrix": data[obs_indices], "obs_indices": obs_indices}]

    def run_discovery(self, n_steps: int = 300, burn_in: int = 50):
        config = MCMCConfig(
            n_steps=n_steps,
            burn_in=burn_in,
            random_state=int(self.rng.integers(0, 2**31 - 1)),
            enable_thermo_constraints=False,
            prior_sparsity=0.2,
            sigma_likelihood=0.2,
            step_size=0.15,
        )
        return SpikeAndSlabSampler(self.engine, config).fit(self.make_dataset())
