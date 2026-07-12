import numpy as np

from pc_mcmc_cigp.discovery import MCMCConfig, MechanismEngine, SpikeAndSlabSampler
from pc_mcmc_cigp.kinetics import ArrheniusRate, MassActionRate, PowerLawRate, ReversibleRate
from pc_mcmc_cigp.reactions import PathwayGenerator, Reaction, Species


def test_engine_supports_mass_action_power_arrhenius_and_reversible_rates():
    a, b, c = Species("A"), Species("B"), Species("C")
    laws = [
        MassActionRate({0: 1.0}),
        PowerLawRate((0,), fixed_orders=(0.5,)),
        ArrheniusRate({0: 1.0}),
        ReversibleRate(MassActionRate({0: 1.0}), MassActionRate({1: 1.0})),
    ]
    for law in laws:
        reaction = Reaction([a], [b], rate_law=law)
        engine = MechanismEngine([a, b, c], [reaction], solver_backend="rk4")
        params = np.mean(engine.rate_parameter_bounds, axis=1)
        if isinstance(law, ArrheniusRate): params = np.array([3.0, 4.0])
        if isinstance(law, ReversibleRate): params = np.array([1.0, 0.1])
        y = engine.simulate(params, [1], [1., 0., 0.], [0., 0.01], temperature=350.)
        assert y.shape == (3, 2) and np.all(np.isfinite(y)) and np.all(y >= 0)


def test_arrhenius_rate_requires_temperature():
    a, b = Species("A"), Species("B")
    engine = MechanismEngine([a, b], [Reaction([a], [b], rate_law=ArrheniusRate({0: 1.0}))])
    try:
        engine.simulate([3., 4.], [1], [1., 0.], [0., 0.1])
    except ValueError as exc:
        assert "temperature" in str(exc)
    else:
        raise AssertionError("missing temperature must be rejected")


def test_pathway_generator_and_group_mcmc_moves_discover_candidate_route():
    a, i, p, side = Species("A"), Species("I"), Species("P"), Species("S")
    reactions = [Reaction([a], [i]), Reaction([i], [p]), Reaction([a], [side])]
    paths = PathwayGenerator([a, i, p, side], reactions).generate(["A"], ["P"], max_steps=3)
    assert [path.reaction_indices for path in paths] == [(0, 1)]
    engine = MechanismEngine([a, i, p, side], reactions)
    t = np.linspace(0, 0.1, 4); y0 = np.array([1., 0., 0., 0.])
    data = engine.simulate([2., 2., 0.1], [1, 1, 0], y0, t)
    result = SpikeAndSlabSampler(engine, MCMCConfig(n_steps=30, burn_in=5, random_state=3, enable_thermo_constraints=False)).fit(
        [{"t": t, "y0_full": y0, "data_matrix": data, "obs_indices": [0, 1, 2, 3]}],
        candidate_pathways=[paths[0].reaction_indices],
    )
    assert result.selected_pathways is not None
    assert result.diagnostics["n_candidate_pathways"] == 1
    assert np.isfinite(result.diagnostics["map_rmse"])
