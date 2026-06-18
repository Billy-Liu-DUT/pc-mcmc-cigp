from __future__ import annotations

import numpy as np

from pc_mcmc_cigp.acquisition import AcquisitionFactory
from pc_mcmc_cigp.cigp import CIGPConfig, CIGPRegressor

R_GAS = 8.314


class EpoxyPhysics:
    param_names = ["log10_A_1", "log10_Ea_1", "log10_A_2", "log10_Ea_2"]

    def __init__(self):
        self.W = np.array([6.0, 4.7, 10.0, 4.9], dtype=float)

    def compute_mean(self, X: np.ndarray, W: np.ndarray) -> np.ndarray:
        out = []
        for styrene, paa, temperature, time_s in np.asarray(X, dtype=float):
            k1 = 10 ** W[0] * np.exp(-(10 ** W[1]) / (R_GAS * temperature))
            k2 = 10 ** W[2] * np.exp(-(10 ** W[3]) / (R_GAS * temperature))
            out.append(_simulate_epoxy(styrene, paa, temperature, time_s, k1, k2))
        return np.asarray(out).reshape(-1, 1)

    def compute_gradients_W(self, X: np.ndarray, W: np.ndarray) -> np.ndarray:
        eps = 1e-4
        grads = []
        for i in range(len(W)):
            wp = W.copy()
            wm = W.copy()
            wp[i] += eps
            wm[i] -= eps
            grads.append(((self.compute_mean(X, wp) - self.compute_mean(X, wm)) / (2 * eps)).ravel())
        return np.column_stack(grads)


class EpoxidationBenchmark:
    """Styrene epoxidation active-learning benchmark."""

    bounds = np.array([[0.8, 1.2], [0.8, 1.2], [303.0, 413.0], [60.0, 3600.0]], dtype=float)

    def __init__(self, random_state: int | None = None, noise_std: float = 0.0) -> None:
        self.rng = np.random.default_rng(random_state)
        self.noise_std = noise_std

    def run_experiment(self, x_phys: np.ndarray) -> tuple[float, float]:
        styrene, paa, temperature, time_s = x_phys
        k1 = 10**6.0 * np.exp(-55_000.0 / (R_GAS * temperature))
        k2 = 10**10.0 * np.exp(-85_000.0 / (R_GAS * temperature))
        y_true = _simulate_epoxy(styrene, paa, temperature, time_s, k1, k2)
        y_obs = max(0.0, y_true + self.rng.normal(0.0, self.noise_std))
        return y_obs, y_true

    def run_optimization(self, n_initial: int = 5, n_iter: int = 10) -> list[dict]:
        X = self._sample_uniform(n_initial)
        y = np.array([self.run_experiment(x)[0] for x in X])
        history = [
            {"iteration": i, "x": X[i].copy(), "yield": float(y[i]), "best_yield": float(np.max(y[: i + 1]))}
            for i in range(n_initial)
        ]
        physics = EpoxyPhysics()

        for i in range(n_iter):
            Xn = self._normalize(X)
            proxy_physics = _NormalizedPhysics(physics, self.bounds)
            model = CIGPRegressor(
                proxy_physics,
                CIGPConfig(max_iters=10, optimize_hyperparameters=False, random_state=int(self.rng.integers(1e9))),
            ).fit(Xn, y)
            candidates = self._sample_uniform(128)
            candidates_n = self._normalize(candidates)
            acquisition = AcquisitionFactory.create("PC_EI", xi=0.01, threshold=0.05, sharpness=5.0)
            scores = acquisition.compute(model, candidates_n, y_best=float(np.max(y)))
            x_next = candidates[int(np.argmax(scores))]
            y_next, _ = self.run_experiment(x_next)
            X = np.vstack([X, x_next])
            y = np.append(y, y_next)
            history.append(
                {
                    "iteration": n_initial + i,
                    "x": x_next.copy(),
                    "yield": float(y_next),
                    "best_yield": float(np.max(y)),
                }
            )
        return history

    def _sample_uniform(self, n: int) -> np.ndarray:
        return self.bounds[:, 0] + self.rng.random((n, self.bounds.shape[0])) * (
            self.bounds[:, 1] - self.bounds[:, 0]
        )

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])


class _NormalizedPhysics:
    def __init__(self, physics: EpoxyPhysics, bounds: np.ndarray) -> None:
        self.physics = physics
        self.bounds = bounds
        self.W = physics.W.copy()
        self.param_names = physics.param_names

    def _inverse(self, X_norm: np.ndarray) -> np.ndarray:
        return self.bounds[:, 0] + X_norm * (self.bounds[:, 1] - self.bounds[:, 0])

    def compute_mean(self, X: np.ndarray, W: np.ndarray) -> np.ndarray:
        return self.physics.compute_mean(self._inverse(X), W)

    def compute_gradients_W(self, X: np.ndarray, W: np.ndarray) -> np.ndarray:
        return self.physics.compute_gradients_W(self._inverse(X), W)


def _simulate_epoxy(styrene: float, paa: float, _temperature: float, time_s: float, k1: float, k2: float) -> float:
    def ode(_t, y):
        s, p, epoxide, acid = np.maximum(y, 0.0)
        r1 = k1 * s * p
        r2 = k2 * epoxide * acid
        return np.array([-r1, -r1, r1 - r2, r1 - r2], dtype=float)

    state = np.array([styrene, paa, 0.0, 0.0], dtype=float)
    n_steps = max(8, min(256, int(time_s / 30)))
    dt = float(time_s) / n_steps
    for step in range(n_steps):
        t = step * dt
        k_1 = ode(t, state)
        k_2 = ode(t + dt / 2, state + dt * k_1 / 2)
        k_3 = ode(t + dt / 2, state + dt * k_2 / 2)
        k_4 = ode(t + dt, state + dt * k_3)
        state = np.maximum(state + dt * (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6, 0.0)
    return float(state[2])
