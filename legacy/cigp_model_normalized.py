# -*- coding: utf-8 -*-
import numpy as np
import GPy
from cigp_model import CIGP
import config  # 导入配置

"""
==================================================================================
v20.0: CIGP 归一化子类 + 软惩罚 (Soft Penalty)
==================================================================================
"""


class CIGP_Normalized(CIGP):
    def __init__(self, X_norm, Y_norm, physics_model, Y_scaler, lambda_penalty=0.0, **kwargs):
        self.Y_scaler = Y_scaler
        self.lambda_penalty = lambda_penalty  # (v20.0) 接收惩罚系数
        super(CIGP_Normalized, self).__init__(X_norm, Y_norm, physics_model, **kwargs)

    def parameters_changed(self):
        # 1. 计算物理均值 (调用 Scipy ODE)
        self.mu_physical = self.physics_model.compute_mean(self.X.values, self.W.values)
        self.mu_physical = np.nan_to_num(self.mu_physical, nan=0.0, posinf=0.0, neginf=0.0)

        # 2. 归一化
        self.mu_total = (self.mu_physical - self.Y_scaler['mean']) / self.Y_scaler['std']

        # 3. 标准 GP 逻辑 (Woodbury identity)
        self.Y_minus_mu = self.Y - self.mu_total
        self.K_err = self.kernel.K(self.X)
        self.sigma_n_sq = self.likelihood.variance
        self.Sigma_total = self.K_err + np.eye(self.X.shape[0]) * self.sigma_n_sq

        try:
            self.L_total = np.linalg.cholesky(self.Sigma_total)
            # alpha = K^{-1} y
            self.woodbury_vector, _ = GPy.util.linalg.dpotrs(self.L_total, self.Y_minus_mu, lower=True)
            self.woodbury_inv, _ = GPy.util.linalg.dpotri(self.L_total, lower=True)

            self.log_det_Sigma = 2.0 * np.sum(np.log(np.diag(self.L_total)))
            self.data_fit = np.dot(self.Y_minus_mu.T, self.woodbury_vector)
        except np.linalg.LinAlgError:
            self.log_det_Sigma = 1e10
            self.data_fit = 1e10

    def log_likelihood(self) -> float:
        # 原始 Log Likelihood (GPy 默认最大化这个)
        # LL = -0.5 * (data_fit + log_det + N*log(2pi))
        # 这里的基类实现似乎返回的是 -NLL (即 LL)
        ll = super(CIGP_Normalized, self).log_likelihood()

        # (v20.0) 加上软惩罚
        # 我们希望最大化 LL，同时最小化 variance。
        # 所以我们要 *减去* (lambda * variance)
        penalty = self.lambda_penalty * self.kernel.variance

        return float(ll - penalty)

    def _param_grad_helper(self, dL_dK):
        # 1. 调用基类计算 W 的梯度 (链式法则)
        super(CIGP_Normalized, self)._param_grad_helper(dL_dK)

        # 2. (v20.0) 手动修改 Kernel Variance 的梯度
        # 目标函数 J = LL - lambda * var
        # dJ/d(var) = d(LL)/d(var) - lambda
        # GPy 已经在 update_gradients_full 里计算了 d(LL)/d(var) 并存在 .gradient 里
        # 我们只需要减去 lambda

        # 注意: 必须先检查梯度是否已经计算 (非 None)
        if self.kernel.variance.gradient is not None:
            self.kernel.variance.gradient -= self.lambda_penalty