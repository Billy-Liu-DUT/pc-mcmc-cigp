import numpy as np

from pc_mcmc_cigp.discovery import (
    MCMCConfig,
    MechanismEngine,
    SpikeAndSlabSampler,
    run_multiple_chains,
)
from pc_mcmc_cigp.discovery.diagnostics import effective_sample_size, gelman_rubin_rhat
from pc_mcmc_cigp.reactions import Reaction, Species


def test_rhat_and_ess_are_finite_for_simple_chains():
    chains = np.array(
        [
            [[0.0, 1.0], [0.1, 0.9], [0.2, 0.8]],
            [[0.05, 0.95], [0.15, 0.85], [0.25, 0.75]],
        ]
    )

    rhat = gelman_rubin_rhat(chains)
    ess = effective_sample_size(chains)

    assert rhat.shape == (2,)
    assert ess.shape == (2,)
    assert np.all(np.isfinite(rhat))
    assert np.all(ess > 0)


def test_run_multiple_chains_merges_pip_and_reports_chain_diagnostics():
    a = Species("A", {"A": 1})
    b = Species("B", {"B": 1})
    engine = MechanismEngine([a, b], [Reaction([a], [b])])
    t = np.linspace(0, 0.2, 4)
    y0 = np.array([1.0, 0.0])
    data = engine.simulate(np.array([1.0]), np.array([1.0]), y0, t)
    dataset = [{"t": t, "y0_full": y0, "data_matrix": data, "obs_indices": [0, 1]}]

    result = run_multiple_chains(
        engine,
        dataset,
        MCMCConfig(n_steps=24, burn_in=4, random_state=10, enable_thermo_constraints=False),
        n_chains=2,
    )

    assert result.posterior_inclusion_probabilities.shape == (1,)
    assert result.diagnostics["n_chains"] == 2
    assert "rhat_max" in result.diagnostics
    assert "ess_min" in result.diagnostics
    assert result.chain_diagnostics["rhat"].shape == (1,)


def test_sampler_keeps_chain_arrays_for_downstream_diagnostics():
    a = Species("A", {"A": 1})
    b = Species("B", {"B": 1})
    engine = MechanismEngine([a, b], [Reaction([a], [b])])
    t = np.linspace(0, 0.1, 3)
    y0 = np.array([1.0, 0.0])
    data = engine.simulate(np.array([1.0]), np.array([1.0]), y0, t)
    dataset = [{"t": t, "y0_full": y0, "data_matrix": data, "obs_indices": [0, 1]}]

    sampler = SpikeAndSlabSampler(
        engine,
        MCMCConfig(n_steps=12, burn_in=2, random_state=11, enable_thermo_constraints=False),
    )
    sampler.fit(dataset)

    assert sampler.chain_z_array.shape == (10, 1)
    assert sampler.chain_theta_array.shape == (10, 1)
