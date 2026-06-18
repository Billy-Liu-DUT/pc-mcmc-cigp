import numpy as np
from scipy.integrate import solve_ivp
from base_physics import BasePhysicsModel
import config

R_GAS = 8.314


def _ode_system_simple(t, C, w_vector_with_T, nu_a, nu_b, nu_p):
    T = w_vector_with_T[-1]
    # 兼容旧参数结构
    alpha, beta, log10_A, log10_Ea = w_vector_with_T[:4]

    A = 10 ** log10_A
    Ea = 10 ** log10_Ea
    k = A * np.exp(-Ea / (R_GAS * np.clip(T, 1e-6, np.inf)))

    C_A, C_B, C_P = C
    C_A = max(0, C_A)
    C_B = max(0, C_B)

    # 动力学方程
    Rate = k * (C_A ** alpha) * (C_B ** beta)

    # 引入化学计量数
    dCA_dt = -nu_a * Rate
    dCB_dt = -nu_b * Rate
    dCP_dt = +nu_p * Rate
    return [dCA_dt, dCB_dt, dCP_dt]


class SimpleReactionODE(BasePhysicsModel):
    def __init__(self, W_init: np.ndarray = None, X_scaler: dict = None, stoichiometry=[1., 1., 1.]):
        if W_init is None: W_init = np.array([1.0, 1.0, 5.0, 4.7])
        self.X_scaler = X_scaler
        self.stoichiometry = stoichiometry
        super().__init__(W_init=W_init, param_names=['alpha', 'beta', 'log10_A', 'log10_Ea'])

    def compute_mean(self, X_norm: np.ndarray, W_log: np.ndarray) -> np.ndarray:
        # ... (保持原有的 solve_ivp 逻辑，记得传入 args=(..., *self.stoichiometry)) ...
        # 为节省篇幅，这里略去重复代码，核心是将 compute_mean 里的 args 改为下面这样：
        # args=(np.append(W_log, Ti), self.stoichiometry[0], self.stoichiometry[1], self.stoichiometry[2])

        # 下面是完整实现代码块，可以直接复制：
        N = X_norm.shape[0]
        mu_pred = np.zeros((N, 1))
        X_physical = self.X_scaler['min'] + X_norm * (self.X_scaler['max'] - self.X_scaler['min'])
        nu_a, nu_b, nu_p = self.stoichiometry

        for i in range(N):
            C_A0, C_B0, Ti, t_final_i = X_physical[i, :]
            t_span = [0, t_final_i]
            w_pack = np.append(W_log, Ti)
            try:
                sol = solve_ivp(_ode_system_simple, t_span, [C_A0, C_B0, 0.0],
                                method=config.ODE_METHOD,
                                args=(w_pack, nu_a, nu_b, nu_p),  # 传入计量数
                                rtol=config.ODE_RTOL, atol=config.ODE_ATOL)
                mu_pred[i, 0] = sol.y[2, -1] if sol.success else 0.0
            except:
                mu_pred[i, 0] = 0.0
        return mu_pred