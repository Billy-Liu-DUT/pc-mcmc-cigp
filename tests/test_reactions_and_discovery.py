import numpy as np

from pc_mcmc_cigp.discovery import MCMCConfig, MechanismEngine, SpikeAndSlabSampler
from pc_mcmc_cigp.reactions import AtomMappedNetworkGenerator, Reaction, Species


def test_generated_reactions_conserve_atoms_and_charge():
    h2 = Species("H2", {"H": 2, "Charge": 0})
    h = Species("H.", {"H": 1, "Charge": 0})

    generator = AtomMappedNetworkGenerator(
        reactants=[h2],
        products=[h],
        fragments=[h],
        max_reactants=2,
        max_products=2,
    )

    reactions = generator.generate()

    assert reactions
    for reaction in reactions:
        assert reaction.is_balanced(generator.atom_types)


def test_engine_detects_reversible_pairs_and_applies_isothermal_balance():
    a = Species("A", {"A": 1})
    b = Species("B", {"B": 1})
    reactions = [Reaction([a], [b]), Reaction([b], [a])]
    engine = MechanismEngine([a, b], reactions)

    rates = engine.calculate_isothermal_rates({"A": 0.0, "B": 1.0}, [2.0, 999.0])

    assert engine.reverse_map == {1: 0}
    assert np.isclose(rates[0], np.exp(2.0))
    assert np.isclose(rates[1], np.exp(2.0) * np.exp(-1.0))


def test_engine_simulate_returns_non_negative_species_by_time_matrix():
    a = Species("A", {"A": 1})
    b = Species("B", {"B": 1})
    engine = MechanismEngine([a, b], [Reaction([a], [b])])

    y = engine.simulate(np.array([1.0]), np.array([1.0]), np.array([1.0, 0.0]), np.linspace(0, 1, 5))

    assert y.shape == (2, 5)
    assert np.all(y >= -1e-9)
    assert y[0, -1] < y[0, 0]
    assert y[1, -1] > y[1, 0]
    assert engine.last_solver_backend in {"scipy", "rk4"}


def test_sampler_fit_returns_discovery_result_with_pip():
    a = Species("A", {"A": 1})
    b = Species("B", {"B": 1})
    engine = MechanismEngine([a, b], [Reaction([a], [b])])
    t = np.linspace(0, 0.2, 4)
    y0 = np.array([1.0, 0.0])
    data = engine.simulate(np.array([1.0]), np.array([1.0]), y0, t)
    dataset = [{"t": t, "y0_full": y0, "data_matrix": data, "obs_indices": [0, 1]}]
    sampler = SpikeAndSlabSampler(
        engine,
        MCMCConfig(n_steps=30, burn_in=5, random_state=4, enable_thermo_constraints=False),
    )

    result = sampler.fit(dataset)

    assert result.posterior_inclusion_probabilities.shape == (1,)
    assert result.mean_parameters.shape == (1,)
    assert result.selected_reactions
    assert result.diagnostics["n_samples"] == 25
    assert "n_active_mean" in result.diagnostics
    assert "map_log_posterior" in result.diagnostics
    assert result.map_structure.shape == (1,)
