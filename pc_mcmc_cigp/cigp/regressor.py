from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CIGPConfig:
    kernel_variance: float = 1.0
    lengthscale: float = 0.5
    noise_variance: float = 1e-4
    kernel_variance_bounds: tuple[float, float] = (1e-8, 10.0)
    lengthscale_bounds: tuple[float, float] = (1e-3, 10.0)
    noise_variance_bounds: tuple[float, float] = (1e-8, 1.0)
    max_iters: int = 200
    random_state: int | None = None
    optimize_hyperparameters: bool = True


class CIGPRegressor:
    """Sklearn-style CIGP wrapper with a physics model as GP prior mean."""

    def __init__(self, physics_model, config: CIGPConfig | None = None) -> None:
        self.physics_model = physics_model
        self.config = config or CIGPConfig()
        self.rng = np.random.default_rng(self.config.random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CIGPRegressor":
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y, dtype=float).reshape(-1, 1)
        if self.X_train_.ndim != 2 or len(self.X_train_) != len(self.y_train_):
            raise ValueError("X must be two-dimensional and contain the same number of rows as y")
        if len(self.X_train_) == 0 or not np.all(np.isfinite(self.X_train_)) or not np.all(np.isfinite(self.y_train_)):
            raise ValueError("training data must be non-empty and finite")
        self.y_mean_ = float(np.mean(self.y_train_))
        self.y_std_ = float(np.std(self.y_train_)) or 1.0
        self.y_norm_ = (self.y_train_ - self.y_mean_) / self.y_std_

        w0 = np.asarray(getattr(self.physics_model, "W", np.zeros(0)), dtype=float)
        if hasattr(self.physics_model, "validate_parameters"):
            self.physics_model.validate_parameters(w0)
        probe = np.asarray(self.physics_model.compute_mean(self.X_train_, w0), dtype=float)
        if probe.shape not in {(len(self.X_train_),), (len(self.X_train_), 1)} or not np.all(np.isfinite(probe)):
            raise ValueError("physics model must return one finite prediction per input row")
        theta0 = np.log(
            [
                self.config.kernel_variance,
                self.config.lengthscale,
                self.config.noise_variance,
            ]
        )
        start = np.concatenate([w0, theta0])

        params = self._coordinate_search(start) if self.config.optimize_hyperparameters else start

        self.W_ = params[: len(w0)]
        if hasattr(self.physics_model, "W"):
            self.physics_model.W = self.W_.copy()
        self.kernel_variance_, self.lengthscale_, self.noise_variance_ = np.exp(params[len(w0) : len(w0) + 3])
        self._refresh_cache()
        return self

    def predict(self, X_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X_new = np.asarray(X_new, dtype=float)
        mu_phys = self._physics_mean(X_new)
        k_star = self._kernel(self.X_train_, X_new)
        residual_mean_norm = k_star.T @ self.alpha_
        mean = mu_phys + residual_mean_norm * self.y_std_
        v = np.linalg.solve(self.K_, k_star)
        var_norm = self._kernel_diag(X_new) - np.sum(k_star * v, axis=0)
        var = np.maximum(var_norm, 1e-12).reshape(-1, 1) * self.y_std_**2 + self.noise_variance_
        return mean.reshape(-1, 1), var

    def predict_residual(self, X_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X_new = np.asarray(X_new, dtype=float)
        k_star = self._kernel(self.X_train_, X_new)
        mean = k_star.T @ self.alpha_ * self.y_std_
        v = np.linalg.solve(self.K_, k_star)
        var_norm = self._kernel_diag(X_new) - np.sum(k_star * v, axis=0)
        return mean.reshape(-1, 1), np.maximum(var_norm, 1e-12).reshape(-1, 1) * self.y_std_**2

    def _negative_log_likelihood(self, params: np.ndarray) -> float:
        n_w = len(np.asarray(getattr(self.physics_model, "W", np.zeros(0)), dtype=float))
        w = params[:n_w]
        if hasattr(self.physics_model, "parameter_bounds"):
            bounds = np.asarray(self.physics_model.parameter_bounds, dtype=float)
            if np.any(w < bounds[:, 0]) or np.any(w > bounds[:, 1]):
                return 1e50
        kernel_variance, lengthscale, noise_variance = np.exp(params[n_w : n_w + 3])
        if not (
            self.config.kernel_variance_bounds[0] <= kernel_variance <= self.config.kernel_variance_bounds[1]
            and self.config.lengthscale_bounds[0] <= lengthscale <= self.config.lengthscale_bounds[1]
            and self.config.noise_variance_bounds[0] <= noise_variance <= self.config.noise_variance_bounds[1]
        ):
            return 1e50
        residual = self.y_norm_ - (self._physics_mean(self.X_train_, w) - self.y_mean_) / self.y_std_
        K = self._kernel(self.X_train_, self.X_train_, kernel_variance, lengthscale)
        K += np.eye(len(self.X_train_)) * noise_variance
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, residual))
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            quadratic = float((residual.T @ alpha).item())
            return 0.5 * quadratic + 0.5 * log_det + 0.5 * len(self.X_train_) * np.log(2 * np.pi)
        except Exception:
            return 1e50

    def _refresh_cache(self) -> None:
        residual = self.y_norm_ - (self._physics_mean(self.X_train_) - self.y_mean_) / self.y_std_
        self.K_ = self._kernel(self.X_train_, self.X_train_) + np.eye(len(self.X_train_)) * self.noise_variance_
        L = np.linalg.cholesky(self.K_)
        self.alpha_ = np.linalg.solve(L.T, np.linalg.solve(L, residual))

    def _coordinate_search(self, start: np.ndarray) -> np.ndarray:
        best = start.copy()
        best_score = self._negative_log_likelihood(best)
        step = 0.25
        for _ in range(max(1, min(self.config.max_iters, 25))):
            improved = False
            for i in range(len(best)):
                for direction in (-1.0, 1.0):
                    candidate = best.copy()
                    candidate[i] += direction * step
                    score = self._negative_log_likelihood(candidate)
                    if score < best_score:
                        best = candidate
                        best_score = score
                        improved = True
            if not improved:
                step *= 0.5
                if step < 1e-3:
                    break
        return best

    def _physics_mean(self, X: np.ndarray, W: np.ndarray | None = None) -> np.ndarray:
        if W is None:
            W = getattr(self, "W_", np.asarray(getattr(self.physics_model, "W", np.zeros(0)), dtype=float))
        return np.asarray(self.physics_model.compute_mean(np.asarray(X, dtype=float), W), dtype=float).reshape(-1, 1)

    def _kernel(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        variance: float | None = None,
        lengthscale: float | None = None,
    ) -> np.ndarray:
        variance = self.kernel_variance_ if variance is None and hasattr(self, "kernel_variance_") else (
            self.config.kernel_variance if variance is None else variance
        )
        lengthscale = self.lengthscale_ if lengthscale is None and hasattr(self, "lengthscale_") else (
            self.config.lengthscale if lengthscale is None else lengthscale
        )
        sq_dist = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
        return variance * np.exp(-0.5 * sq_dist / (lengthscale**2))

    def _kernel_diag(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self.kernel_variance_, dtype=float)
