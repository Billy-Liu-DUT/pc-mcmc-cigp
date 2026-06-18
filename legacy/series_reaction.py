import numpy as np
from scipy.integrate import solve_ivp
from base_physics import BasePhysicsModel
import config

R_GAS = 8.314


def _ode_series_system(t, C, w_vector_with_T, stoich_step1, stoich_step2):
    """
    连串反应 ODE:
    Step 1: nu_a1*A + nu_b1*B -> nu_p1*P  (Rate1)
    Step 2: nu_p2*P           -> nu_s2*S  (Rate2)
    """
    # 1. 提取温度和参数
    T = w_vector_with_T[-1]
    params = w_vector_with_T[:-1]

    # 参数结构 (7个): [alpha1, beta1, logA1, logE1, alpha2, logA2, logE2]
    # Step 1 (A+B->P)
    alpha1, beta1, logA1, logE1 = params[0:4]
    # Step 2 (P->S, 假设只与P有关)
    alpha2, logA2, logE2 = params[4:7]

    # 2. 计算速率常数
    k1 = (10 ** logA1) * np.exp(-(10 ** logE1) / (R_GAS * np.clip(T, 1e-6, np.inf)))
    k2 = (10 ** logA2) * np.exp(-(10 ** logE2) / (R_GAS * np.clip(T, 1e-6, np.inf)))

    # 3. 浓度处理
    C_A, C_B, C_P, C_S = C
    C_A = max(0, C_A)
    C_B = max(0, C_B)
    C_P = max(0, C_P)

    # 4. 计算速率
    # Rate 1: 生成 P
    r1 = k1 * (C_A ** alpha1) * (C_B ** beta1)
    # Rate 2: 消耗 P
    r2 = k2 * (C_P ** alpha2)

    # 5. 微分方程
    # 解包化学计量数
    nu_a1, nu_b1, nu_p1 = stoich_step1  # [1, 1, 1]
    nu_p2, nu_s2 = stoich_step2  # [1, 1] (P -> S)

    dCA = -nu_a1 * r1
    dCB = -nu_b1 * r1
    # P 的净速率 = (生成) - (消耗)
    dCP = +nu_p1 * r1 - nu_p2 * r2
    dCS = +nu_s2 * r2

    return [dCA, dCB, dCP, dCS]


class SeriesReactionODE(BasePhysicsModel):
    """
    连串反应算子 (A+B -> P -> S)
    包含 7 个优化参数。
    """

    def __init__(self, W_init=None, X_scaler=None,
                 stoichiometry_step1=[1., 1., 1.],
                 stoichiometry_step2=[1., 1.]):

        self.X_scaler = X_scaler
        self.stoich1 = stoichiometry_step1
        self.stoich2 = stoichiometry_step2

        # 定义 7 个参数名
        param_names = ['alpha_1', 'beta_1', 'log10_A_1', 'log10_Ea_1',
                       'alpha_2', 'log10_A_2', 'log10_Ea_2']

        # 默认初始值
        if W_init is None:
            # Step 1 快, Step 2 慢
            W_init = np.array([1.0, 1.0, 5.0, 4.5,  # Step 1
                               1.0, 4.0, 4.8])  # Step 2

        super().__init__(W_init, param_names)

    def compute_mean(self, X_norm, W_log):
        N = X_norm.shape[0]
        mu_pred = np.zeros((N, 1))
        X_phys = self.X_scaler['min'] + X_norm * (self.X_scaler['max'] - self.X_scaler['min'])

        for i in range(N):
            C_A0, C_B0, Ti, t_final = X_phys[i, :]
            # [A, B, P, S]
            y0 = [C_A0, C_B0, 0.0, 0.0]
            w_pack = np.append(W_log, Ti)

            try:
                sol = solve_ivp(_ode_series_system, [0, t_final], y0,
                                method=config.ODE_METHOD,
                                args=(w_pack, self.stoich1, self.stoich2),
                                rtol=config.ODE_RTOL, atol=config.ODE_ATOL)
                if sol.success:
                    # 输出中间产物 P
                    mu_pred[i, 0] = sol.y[2, -1]
                else:
                    mu_pred[i, 0] = 0.0
            except:
                mu_pred[i, 0] = 0.0
        return mu_pred