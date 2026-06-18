import numpy as np
from scipy.integrate import solve_ivp
from base_physics import BasePhysicsModel
import config

R_GAS = 8.314  # J/(mol K)


def _ode_reversible_system(t, C, w_vector_with_T, nu_a, nu_b, nu_p):
    """
    通用可逆反应 ODE 系统:
    化学式: nu_a*A + nu_b*B <--> nu_p*P

    正向速率: Rate_fwd = k_f * A^alpha * B^beta
    逆向速率: Rate_rev = k_r * P^gamma
    净速率:   Rate_net = Rate_fwd - Rate_rev
    """
    # 1. 提取温度 (w_vector_with_T 的最后一个元素)
    T = w_vector_with_T[-1]
    params = w_vector_with_T[:-1]

    # 2. 动态解析参数
    # 根据参数数量判断是否包含反应级数 (alpha, beta, gamma)
    if len(params) == 4:
        # 模式 A: 固定级数 (基元反应假设)
        # 参数顺序: [logA_f, logE_f, logA_r, logE_r]
        alpha, beta, gamma = 1.0, 1.0, 1.0
        log10_A_fwd, log10_Ea_fwd, log10_A_rev, log10_Ea_rev = params
    elif len(params) == 7:
        # 模式 B: 可变级数 (数据驱动优化)
        # 参数顺序: [alpha, beta, gamma, logA_f, logE_f, logA_r, logE_r]
        alpha, beta, gamma, log10_A_fwd, log10_Ea_fwd, log10_A_rev, log10_Ea_rev = params
    else:
        # 防御性编程: 如果参数长度不对，抛出错误
        raise ValueError(f"ODE 参数长度错误: 期望 4 或 7，实际得到 {len(params)}")

    # 3. 计算速率常数 (Arrhenius 方程)
    # 使用 np.clip 防止温度为 0 或负数导致溢出
    k_fwd = (10 ** log10_A_fwd) * np.exp(-(10 ** log10_Ea_fwd) / (R_GAS * np.clip(T, 1e-6, np.inf)))
    k_rev = (10 ** log10_A_rev) * np.exp(-(10 ** log10_Ea_rev) / (R_GAS * np.clip(T, 1e-6, np.inf)))

    # 4. 解析浓度并进行物理截断 (防止负浓度)
    C_A, C_B, C_P = C
    C_A = max(0, C_A)
    C_B = max(0, C_B)
    C_P = max(0, C_P)

    # 5. 计算反应速率
    rate_fwd = k_fwd * (C_A ** alpha) * (C_B ** beta)
    rate_rev = k_rev * (C_P ** gamma)
    rate_net = rate_fwd - rate_rev

    # 6. 微分方程 (结合化学计量数)
    # dC/dt = +/- coeff * Rate_net
    dCA_dt = -nu_a * rate_net
    dCB_dt = -nu_b * rate_net
    dCP_dt = +nu_p * rate_net

    return [dCA_dt, dCB_dt, dCP_dt]


class ReversibleReactionODE(BasePhysicsModel):
    """
    通用可逆反应算子 (Reversible Reaction Operator)
    支持:
    1. 自定义化学计量数 (stoichiometry)
    2. 可变/固定反应级数 (variable_orders)
    """

    def __init__(self, W_init: np.ndarray = None,
                 X_scaler: dict = None,
                 variable_orders: bool = True,  # 默认开启级数优化
                 stoichiometry: list = None):  # 默认 [1, 1, 1]

        self.variable_orders = variable_orders
        self.X_scaler = X_scaler

        # 处理化学计量数 [nu_a, nu_b, nu_p]
        if stoichiometry is None:
            self.stoichiometry = [1.0, 1.0, 1.0]
        else:
            if len(stoichiometry) != 3:
                raise ValueError("stoichiometry 必须包含 3 个元素 [nu_a, nu_b, nu_p]")
            self.stoichiometry = stoichiometry

        if self.X_scaler is None:
            raise ValueError("必须传入 X_scaler 以进行物理量反归一化")

        # --- 定义参数名和默认值 ---
        if not self.variable_orders:
            # 模式 A: 固定级数 (4参数)
            expected_names = ['log10_A_fwd', 'log10_Ea_fwd', 'log10_A_rev', 'log10_Ea_rev']
            default_W = np.array([5.0, 4.5, 3.0, 4.8])
        else:
            # 模式 B: 可变级数 (7参数)
            expected_names = ['alpha', 'beta', 'gamma',
                              'log10_A_fwd', 'log10_Ea_fwd', 'log10_A_rev', 'log10_Ea_rev']
            # 默认初始值: 级数初始设为 1.0
            default_W = np.array([1.0, 1.0, 1.0, 5.0, 4.5, 3.0, 4.8])

        # 如果用户没有提供 W_init，使用默认值
        if W_init is None:
            W_init = default_W

        # 验证 W_init 长度
        if len(W_init) != len(expected_names):
            raise ValueError(
                f"W_init 长度 ({len(W_init)}) 与配置不匹配 (variable_orders={variable_orders})。需要 {len(expected_names)} 个参数。")

        super().__init__(W_init=W_init, param_names=expected_names)

    def compute_mean(self, X_norm: np.ndarray, W_log: np.ndarray) -> np.ndarray:
        """
        计算物理模型预测值 (产物浓度 P)
        """
        N = X_norm.shape[0]
        mu_pred = np.zeros((N, 1))

        # 反归一化输入 X -> (CA0, CB0, T, t)
        X_physical = self.X_scaler['min'] + X_norm * (self.X_scaler['max'] - self.X_scaler['min'])

        # 获取化学计量数
        nu_a, nu_b, nu_p = self.stoichiometry

        for i in range(N):
            C_A0, C_B0, Ti, t_final_i = X_physical[i, :]

            # 初始浓度 [CA, CB, CP] (假设产物初始为0)
            C_init = [C_A0, C_B0, 0.0]
            t_span = [0, t_final_i]

            # 拼接参数和温度: [...W..., T]
            w_with_T = np.append(W_log, Ti)

            try:
                sol = solve_ivp(
                    fun=_ode_reversible_system,
                    t_span=t_span,
                    y0=C_init,
                    method=config.ODE_METHOD,  # 使用 config 配置
                    # 将化学计量数作为 args 传入 ODE
                    args=(w_with_T, nu_a, nu_b, nu_p),
                    rtol=config.ODE_RTOL,
                    atol=config.ODE_ATOL
                )

                if sol.success:
                    C_P_final = sol.y[2, -1]
                    # 结果检查 (防止数值爆炸导致的 NaN/Inf)
                    if not np.isfinite(C_P_final):
                        mu_pred[i, 0] = 0.0
                    else:
                        mu_pred[i, 0] = C_P_final
                else:
                    mu_pred[i, 0] = 0.0

            except Exception:
                mu_pred[i, 0] = 0.0

        return mu_pred