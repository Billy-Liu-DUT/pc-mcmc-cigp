import numpy as np
from scipy.integrate import solve_ivp
from base_physics import BasePhysicsModel
import config

R_GAS = 8.314


def _ode_parallel_system(t, C, w_vector, nu_list):
    """
    平行反应 ODE:
    1. Main: nu_a1*A + nu_b1*B -> nu_p*P  (Rate1)
    2. Side: nu_a2*A + nu_b2*B -> nu_s*S  (Rate2)
    """
    # 1. 提取温度
    T = w_vector[-1]
    params = w_vector[:-1]

    # 解包化学计量数 [[nu_a1, nu_b1, nu_p], [nu_a2, nu_b2, nu_s]]
    (nu_a1, nu_b1, nu_p), (nu_a2, nu_b2, nu_s) = nu_list

    # 2. 动态解析参数
    # 如果参数是 4 个: 固定级数 (A1, E1, A2, E2)
    # 如果参数是 8 个: 优化级数 (a1, b1, A1, E1, a2, b2, A2, E2)

    if len(params) == 4:
        # Fixed Orders (Defaults to 1.0)
        alpha1, beta1 = 1.0, 1.0
        alpha2, beta2 = 1.0, 1.0
        logA1, logE1, logA2, logE2 = params
    elif len(params) == 8:
        # Variable Orders
        alpha1, beta1, logA1, logE1, alpha2, beta2, logA2, logE2 = params
    else:
        raise ValueError(f"Parallel ODE 参数长度错误: {len(params)}")

    # 3. 计算速率常数 k1, k2
    k1 = (10 ** logA1) * np.exp(-(10 ** logE1) / (R_GAS * np.clip(T, 1e-6, np.inf)))
    k2 = (10 ** logA2) * np.exp(-(10 ** logE2) / (R_GAS * np.clip(T, 1e-6, np.inf)))

    # 4. 浓度处理
    C_A, C_B, C_P, C_S = C
    C_A = max(0, C_A)
    C_B = max(0, C_B)

    # 5. 计算速率
    r1 = k1 * (C_A ** alpha1) * (C_B ** beta1)
    r2 = k2 * (C_A ** alpha2) * (C_B ** beta2)

    # 6. 微分方程
    # A 和 B 同时被两个反应消耗
    dCA = -nu_a1 * r1 - nu_a2 * r2
    dCB = -nu_b1 * r1 - nu_b2 * r2
    dCP = +nu_p * r1
    dCS = +nu_s * r2

    return [dCA, dCB, dCP, dCS]


class ParallelReactionODE(BasePhysicsModel):
    """
    平行竞争反应算子 (A+B->P, A+B->S)
    """

    def __init__(self, W_init=None, X_scaler=None,
                 variable_orders=True,
                 stoichiometry_main=[1., 1., 1.],  # [A, B, P]
                 stoichiometry_side=[1., 1., 1.]):  # [A, B, S]

        self.X_scaler = X_scaler
        self.variable_orders = variable_orders
        self.stoich_list = [stoichiometry_main, stoichiometry_side]

        if not variable_orders:
            # 模式 A: 固定级数 (4参数)
            names = ['log10_A_main', 'log10_Ea_main', 'log10_A_side', 'log10_Ea_side']
            default_W = np.array([5.0, 4.5, 4.0, 4.8])
        else:
            # 模式 B: 可变级数 (8参数)
            names = ['alpha_main', 'beta_main', 'log10_A_main', 'log10_Ea_main',
                     'alpha_side', 'beta_side', 'log10_A_side', 'log10_Ea_side']
            default_W = np.array([1., 1., 5.0, 4.5, 1., 1., 4.0, 4.8])

        if W_init is None: W_init = default_W
        super().__init__(W_init, names)

    def compute_mean(self, X_norm, W_log):
        N = X_norm.shape[0]
        mu_pred = np.zeros((N, 1))
        X_phys = self.X_scaler['min'] + X_norm * (self.X_scaler['max'] - self.X_scaler['min'])

        for i in range(N):
            C_A0, C_B0, Ti, t_final = X_phys[i, :]
            # 初始状态: [A, B, P, S]
            y0 = [C_A0, C_B0, 0.0, 0.0]
            w_pack = np.append(W_log, Ti)

            try:
                sol = solve_ivp(_ode_parallel_system, [0, t_final], y0,
                                method=config.ODE_METHOD,
                                args=(w_pack, self.stoich_list),
                                rtol=config.ODE_RTOL, atol=config.ODE_ATOL)
                if sol.success:
                    # 我们只输出主产物 P 的浓度用于拟合
                    mu_pred[i, 0] = sol.y[2, -1]
                else:
                    mu_pred[i, 0] = 0.0
            except:
                mu_pred[i, 0] = 0.0
        return mu_pred