import numpy as np
from scipy.integrate import solve_ivp
from base_physics import BasePhysicsModel
import config_benchmark as cfg  # 使用新的 config

R_GAS = 8.314


def _ode_epoxy_system(t, C, w_vector_with_T):
    """
    苯乙烯环氧化 ODE (自毒化机理):
    """
    T = w_vector_with_T[-1]
    params = w_vector_with_T[:-1]

    # 解析参数 (8个)
    alpha1, beta1, logA1, logE1 = params[0:4]
    alpha2, beta2, logA2, logE2 = params[4:8]

    # 速率常数
    T_safe = np.clip(T, 1e-6, np.inf)
    k1 = (10 ** logA1) * np.exp(-(10 ** logE1) / (R_GAS * T_safe))
    k2 = (10 ** logA2) * np.exp(-(10 ** logE2) / (R_GAS * T_safe))

    C_A, C_B, C_P, C_D = C
    # 截断负浓度
    C_A, C_B, C_P, C_D = [max(0, x) for x in [C_A, C_B, C_P, C_D]]

    # 速率方程
    r1 = k1 * (C_A ** alpha1) * (C_B ** beta1)
    r2 = k2 * (C_P ** alpha2) * (C_D ** beta2)  # 依赖乙酸 D

    # 质量平衡
    dCA = -r1
    dCB = -r1
    dCP = r1 - r2
    dCD = r1 - r2

    return [dCA, dCB, dCP, dCD]


class EpoxyReactionODE(BasePhysicsModel):
    def __init__(self, W_init=None, X_scaler=None):
        self.X_scaler = X_scaler
        param_names = [
            'alpha_1', 'beta_1', 'log10_A_1', 'log10_Ea_1',
            'alpha_2', 'beta_2', 'log10_A_2', 'log10_Ea_2'
        ]
        # 初始猜测
        if W_init is None:
            W_init = np.array([1.0, 1.0, 6.0, 5.0, 1.0, 1.0, 9.0, 5.2])
        super().__init__(W_init, param_names)

    def compute_mean(self, X_norm, W_log):
        N = X_norm.shape[0]
        mu_pred = np.zeros((N, 1))
        # 反归一化
        X_phys = self.X_scaler['min'] + X_norm * (self.X_scaler['max'] - self.X_scaler['min'])

        for i in range(N):
            C_A0, C_B0, Ti, t_final = X_phys[i, :]
            y0 = [C_A0, C_B0, 0.0, 0.0]  # [A, B, P, D]
            w_pack = np.append(W_log, Ti)

            try:
                sol = solve_ivp(_ode_epoxy_system, [0, t_final], y0,
                                method='LSODA', args=(w_pack,),
                                rtol=1e-3, atol=1e-6)  # 适当放宽精度以加速BO
                mu_pred[i, 0] = sol.y[2, -1] if sol.success else 0.0
            except:
                mu_pred[i, 0] = 0.0
        return mu_pred