# -*- coding: utf-8 -*-
import numpy as np
from scipy.integrate import solve_ivp
from base_physics import BasePhysicsModel
import config_robertson as cfg


class RobertsonGeneralized(BasePhysicsModel):
    def __init__(self, mode='variable', scale_factor=1.0, w_init=None):
        self.mode = mode

        # [FIX 1] 补回缺失的 scale_factor 定义，解决 AttributeError
        self.scale_factor = scale_factor

        # [FIX 2] 针对 mmol/L 单位修正真值
        # 解释：因为浓度放大了1000倍，二级反应的k必须缩小1000倍 (log减3)
        self.truth_params_phys = np.array([
            -0.4,  # log_k1 (一级，不变)
            5.48,  # log_k2 (二级，原值8.48 - 3.0)
            2.0,  # log_k3 (二级，原值5.0 - 3.0)
            -8.0,  # log_k4 (一级，不变)
            1.0, 2.0, 1.0, 1.0  # 级数
        ])

        names = cfg.CURRENT_PARAM_NAMES
        if w_init is None:
            w_init_norm = np.ones(len(names)) * 0.5
        else:
            w_init_norm = w_init

        bounds_dict = cfg.W_BOUNDS
        self.lb = np.array([bounds_dict[name][0] for name in names])
        self.ub = np.array([bounds_dict[name][1] for name in names])
        self.diff = self.ub - self.lb

        super().__init__(w_init_norm, names)

    def _unscale_params(self, W_norm):
        return self.lb + W_norm * self.diff

    def _ode_system(self, t, y, params_phys):
        A, B, C = y
        # 保护：防止浓度为负导致指数运算错误
        A = max(A, 1e-12)
        B = max(B, 1e-12)
        C = max(C, 1e-12)

        log_k = params_phys[:4]
        n = params_phys[4:]
        k1, k2, k3, k4 = 10 ** log_k
        n1, n2, n3, n4 = n

        # 速率计算
        r1 = k1 * (A ** n1)
        r2 = k2 * (B ** n2)
        r3 = k3 * (B ** n3) * C
        r4 = k4 * (C ** n4)

        dA = -r1 + r3
        dB = r1 - r2 - r3
        dC = r2 - r4
        return [dA, dB, dC]

    def compute_mean(self, X, W_norm):
        W_phys = self._unscale_params(W_norm)
        t_eval = X.flatten()
        idx_sort = np.argsort(t_eval)
        t_sorted = t_eval[idx_sort]

        if len(t_sorted) == 0: return np.zeros((len(X), 1))

        t_ode = t_sorted.copy()
        if t_sorted[0] > 1e-9:
            t_ode = np.insert(t_ode, 0, 0.0)

        # 初始浓度 mmol/L
        y0 = cfg.DATA_GEN_CONFIG['INITIAL_CONC']

        # [CRITICAL] 使用 Radau 防止刚性爆炸
        try:
            sol = solve_ivp(
                lambda t, y: self._ode_system(t, y, W_phys),
                [0, np.max(t_ode) + 1.0],
                y0,
                t_eval=t_ode,
                method='Radau',
                rtol=1e-6,
                atol=1e-9
            )

            if not sol.success:
                return np.zeros((len(X), 1))

            y_pred = sol.y[2, :]
            if t_sorted[0] > 1e-9:
                y_pred = y_pred[1:]

            y_final = np.zeros_like(y_pred)
            y_final[idx_sort] = y_pred

            return y_final.reshape(-1, 1) * self.scale_factor

        except Exception as e:
            print(f"[ODE Error] {e}")
            return np.zeros((len(X), 1))