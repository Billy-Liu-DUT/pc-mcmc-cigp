import numpy as np
from abc import ABC, abstractmethod


class BasePhysicsModel(ABC):

    def __init__(self, W_init: np.ndarray, param_names: list[str] = None):
        self._W = W_init
        self.P = len(self._W)
        if param_names and len(param_names) != self.P:
            raise ValueError(f"param_names 长度 ({len(param_names)}) 必须等于 W_init 长度 ({self.P})")
        self.param_names = param_names or [f"w_{i}" for i in range(self.P)]

    @property
    def W(self) -> np.ndarray:
        return self._W

    @W.setter
    def W(self, value: np.ndarray):
        if value.shape != (self.P,):
            raise ValueError(f"W 的形状必须是 ({self.P},)")
        self._W = value

    @abstractmethod
    def compute_mean(self, X: np.ndarray, W: np.ndarray) -> np.ndarray:
        pass

    def compute_gradients_W(self, X: np.ndarray, W: np.ndarray) -> np.ndarray:
        """
        使用中心差分法计算均值对 W 的雅可比矩阵 d(mu_total) / dW。
        """
        N = X.shape[0]
        P = W.shape[0]
        dmu_dW = np.zeros((N, P))

        epsilon_abs = 1e-6  # 绝对步长
        epsilon_rel = 1e-4  # 相对步长

        for i in range(P):
            h = epsilon_abs + epsilon_rel * abs(W[i])

            # 扰动 W[i] + h
            W_plus = W.copy()
            W_plus[i] += h
            mu_plus = self.compute_mean(X, W_plus)

            # 扰动 W[i] - h
            W_minus = W.copy()
            W_minus[i] -= h

            # 【关键 Bug 修复 v11.0】: 确保使用 W_minus
            mu_minus = self.compute_mean(X, W_minus)

            # 中心差分
            grad_col = (mu_plus - mu_minus) / (2 * h)

            # 【关键修正】强制将任何 nan/inf 梯度转换为 0
            dmu_dW[:, i] = np.nan_to_num(grad_col.flatten(), nan=0.0, posinf=0.0, neginf=0.0)

        return dmu_dW