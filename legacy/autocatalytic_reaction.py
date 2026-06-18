import numpy as np
from scipy.integrate import solve_ivp
from base_physics import BasePhysicsModel
import config

R_GAS = 8.314


def _ode_autocat_system(t, C, w_vector_with_T, stoichiometry):
    """
    自催化 ODE: nu_a*A + nu_p_react*P -> nu_p_prod*P
    Rate = k * A^alpha * P^beta
    """
    T = w_vector_with_T[-1]
    alpha, beta, logA, logE = w_vector_with_T[:-1]

    k = (10 ** logA) * np.exp(-(10 ** logE) / (R_GAS * np.clip(T, 1e-6, np.inf)))

    C_A, C_P = C
    C_A = max(0, C_A)
    C_P = max(0, C_P)

    # 【关键】自催化启动机制
    # 如果 P 完全为 0，反应无法启动。假设有极微量 P 存在，或由非催化路径启动。
    # 这里使用 epsilon 技巧
    C_P_effective = max(C_P, 1e-9)

    Rate = k * (C_A ** alpha) * (C_P_effective ** beta)

    # 化学计量数: [nu_a, nu_p_react, nu_p_prod]
    # A + P -> 2P  => nu_a=1, nu_p_in=1, nu_p_out=2
    nu_a, nu_p_in, nu_p_out = stoichiometry

    dCA = -nu_a * Rate
    dCP = -nu_p_in * Rate + nu_p_out * Rate  # 净生成

    return [dCA, dCP]


class AutocatalyticODE(BasePhysicsModel):
    """
    自催化反应算子 (A + P -> 2P)
    """

    def __init__(self, W_init=None, X_scaler=None,
                 stoichiometry=[1., 1., 2.]):  # 默认 A+P->2P

        self.X_scaler = X_scaler
        self.stoichiometry = stoichiometry

        param_names = ['alpha', 'beta', 'log10_A', 'log10_Ea']

        if W_init is None:
            # 自催化通常级数较高，这里设初值为 1.0
            W_init = np.array([1.0, 1.0, 5.0, 4.5])

        super().__init__(W_init, param_names)

    def compute_mean(self, X_norm, W_log):
        N = X_norm.shape[0]
        mu_pred = np.zeros((N, 1))
        X_phys = self.X_scaler['min'] + X_norm * (self.X_scaler['max'] - self.X_scaler['min'])

        nu_list = self.stoichiometry

        for i in range(N):
            C_A0, C_B0, Ti, t_final = X_phys[i, :]
            # 注意：自催化模型通常不依赖 B，或者 B 是过量的。
            # 这里我们只追踪 [A, P]。假设初始 P=0 (靠 eps 启动)
            y0 = [C_A0, 0.0]
            w_pack = np.append(W_log, Ti)

            try:
                # 推荐使用 LSODA，因为自催化爆发期非常刚性
                sol = solve_ivp(_ode_autocat_system, [0, t_final], y0,
                                method='LSODA',
                                args=(w_pack, nu_list),
                                rtol=config.ODE_RTOL, atol=config.ODE_ATOL)
                if sol.success:
                    mu_pred[i, 0] = sol.y[1, -1]  # P
                else:
                    mu_pred[i, 0] = 0.0
            except:
                mu_pred[i, 0] = 0.0
        return mu_pred