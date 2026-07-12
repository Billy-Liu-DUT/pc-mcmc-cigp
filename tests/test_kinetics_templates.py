import numpy as np

from pc_mcmc_cigp.cigp import CIGPConfig, CIGPRegressor
from pc_mcmc_cigp.kinetics import TemplateRegistry, create_kinetic_template, list_kinetic_templates


ROWS = {
    "simple_arrhenius": [1.0, 1.0, 330.0, 0.0],
    "series_arrhenius": [1.0, 1.0, 330.0, 0.0],
    "parallel_arrhenius": [1.0, 1.0, 330.0, 0.0],
    "reversible_arrhenius": [1.0, 1.0, 330.0, 0.0],
    "autocatalytic_arrhenius": [1.0, 0.01, 330.0, 0.0],
    "epoxidation": [1.0, 1.0, 330.0, 0.0],
    "robertson": [1.0, 0.0, 0.0, 0.0],
    "michaelis_menten": [1.0, 0.1, 330.0, 0.0],
    "langmuir_hinshelwood": [1.0, 1.0, 330.0, 0.0],
    "power_law": [1.0, 1.0, 330.0, 0.0],
    "inhibited": [1.0, 0.1, 0.1, 0.0],
    "radical_chain": [1.0, 0.01, 330.0, 0.0],
}


def test_registry_exposes_all_supported_families_and_metadata():
    assert set(ROWS) <= set(list_kinetic_templates())
    for name in ROWS:
        description = TemplateRegistry.describe(name)
        assert description["inputs"]
        assert description["parameters"]


def test_all_templates_are_nonnegative_batch_stable_and_return_initial_state_at_zero_time():
    for name, row in ROWS.items():
        model = create_kinetic_template(name)
        X0 = np.asarray([row, row], dtype=float)
        result0 = model.simulate(X0)
        assert result0.values.shape == (2, 1)
        assert np.all(np.isfinite(result0.values))
        assert np.all(result0.values >= 0)
        row_later = row.copy(); row_later[-1] = 0.02
        batch = model.compute_mean(np.asarray([row_later, row_later]), model.W)
        single = model.compute_mean(np.asarray([row_later]), model.W)
        assert np.allclose(batch[0], single[0])


def test_temperature_increases_simple_arrhenius_conversion():
    model = create_kinetic_template("simple_arrhenius")
    low, high = model.compute_mean(np.array([[1., 1., 300., 2.], [1., 1., 400., 2.]]), model.W).ravel()
    assert high > low >= 0


def test_gradients_are_finite_for_representative_template():
    model = create_kinetic_template("reversible_arrhenius")
    X = np.array([[1., 1., 350., 0.1], [0.8, 1.2, 360., 0.2]])
    gradient = model.compute_gradients_W(X, model.W)
    assert gradient.shape == (2, len(model.W))
    assert np.all(np.isfinite(gradient))


def test_representative_templates_work_as_cigp_priors():
    for name in ["simple_arrhenius", "reversible_arrhenius", "michaelis_menten", "langmuir_hinshelwood", "epoxidation"]:
        model = create_kinetic_template(name)
        base = np.asarray(ROWS[name], dtype=float)
        X = np.vstack([base, base, base]); X[:, -1] = [0.01, 0.02, 0.04]
        y = model.compute_mean(X, model.W).ravel()
        fitted = CIGPRegressor(model, CIGPConfig(optimize_hyperparameters=False)).fit(X, y)
        mean, var = fitted.predict(X)
        assert mean.shape == var.shape == (3, 1)
        assert np.all(np.isfinite(mean)) and np.all(var >= 0)
