import numpy as np

from pc_mcmc_cigp.acquisition import AcquisitionFactory
from pc_mcmc_cigp.cigp import CIGPConfig, CIGPRegressor


class LinearPhysics:
    param_names = ["slope", "intercept"]

    def __init__(self):
        self.W = np.array([0.5, 0.0], dtype=float)

    def compute_mean(self, X, W):
        return (W[0] * X[:, :1] + W[1]).reshape(-1, 1)

    def compute_gradients_W(self, X, W):
        return np.column_stack([X[:, 0], np.ones(X.shape[0])])


def test_cigp_regressor_fit_predict_and_residual_shapes_are_stable():
    X = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
    y = 2.0 * X[:, 0] + 0.1
    model = CIGPRegressor(LinearPhysics(), CIGPConfig(max_iters=5, random_state=7))

    fitted = model.fit(X, y)
    mean, var = fitted.predict(X[:3])
    residual_mean, residual_var = fitted.predict_residual(X[:3])

    assert fitted is model
    assert mean.shape == (3, 1)
    assert var.shape == (3, 1)
    assert residual_mean.shape == (3, 1)
    assert residual_var.shape == (3, 1)
    assert np.all(np.isfinite(mean))
    assert np.all(var >= 0)


def test_all_acquisition_functions_return_finite_scores():
    X = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
    y = 2.0 * X[:, 0] + 0.1
    model = CIGPRegressor(LinearPhysics(), CIGPConfig(max_iters=3, random_state=8)).fit(X, y)
    candidates = np.linspace(0.0, 1.0, 5).reshape(-1, 1)

    for name in ["EI", "GWU", "DH", "PC_EI"]:
        acquisition = AcquisitionFactory.create(name, beta=0.5, threshold=0.1, xi=0.01)
        scores = acquisition.compute(model, candidates, y_best=float(np.max(y)))
        assert scores.shape == (5,)
        assert np.all(np.isfinite(scores))
