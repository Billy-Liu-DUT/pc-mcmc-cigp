# -*- coding: utf-8 -*-
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import qmc
import config

"""
==================================================================================
v17.1 Hotfix:
- (Fix) 增加了 v10.0 重构时遗漏的 R_GAS (8.314) 全局常量。

v17.0 更新:
- (ODE) "真实" ODE 已从 "竞争" (v10) 切换为 "产物分解" (v16.0)。
- (Params) "真实" W_MAIN/W_SIDE 参数已更新为 v16.0 调谐结果。
- (Range) "真实" T/t 范围已更新为 v16.0 调谐结果。
==================================================================================
"""

# (v17.1 Hotfix) 增加缺失的常量
R_GAS = 8.314


# --- 1. 【v17.0】 “真实”的产物分解 ODE (来自 v16.0 调谐器) ---
def _ode_system_DECOMPOSITION(t, C, w_main, w_side, T):
    """
    v16.0 "真实" ODE: 产物分解
    - 反应 1: A + B -> P  (k1, w_main)
    - 反应 2: P   -> S  (k2, w_side)
    """
    # 反应 1 (Main)
    alpha1, beta1, A1, Ea1 = w_main
    # 反应 2 (Side) - 假设 P -> S 是一级反应 (alpha2=1.0)
    alpha2_side, _, A2, Ea2 = w_side

    k1 = A1 * np.exp(-Ea1 / (R_GAS * np.clip(T, 1e-6, np.inf)))
    k2 = A2 * np.exp(-Ea2 / (R_GAS * np.clip(T, 1e-6, np.inf)))

    C_A, C_B, C_P, C_S = C

    C_A = max(0, C_A)
    C_B = max(0, C_B)
    C_P = max(0, C_P)

    Rate1 = k1 * (C_A ** alpha1) * (C_B ** beta1)
    Rate2 = k2 * (C_P ** alpha2_side)  # P 的分解

    dCA_dt = -Rate1
    dCB_dt = -Rate1
    dCP_dt = +Rate1 - Rate2  # P 被生成，同时被消耗
    dCS_dt = +Rate2

    return [dCA_dt, dCB_dt, dCP_dt, dCS_dt]


# --- 2. 【v17.0 更新】 生成“非凸”模拟数据 ---
def generate_synthetic_data(N: int = 20):
    print(f"--- 正在生成 {N} 条“真实”模拟数据 (v17.0 “产物分解”模型)... ---")

    # 2.1 "真实"物理参数 (来自 v16.0 调谐器)
    w_true_P = np.array([
        1.0, 1.0,  # (alpha1, beta1)
        5e6,  # A1
        5.5e4  # Ea1
    ])
    w_true_S = np.array([
        1.0, 0.0,  # (alpha2_side, beta2_unused)
        1e6,  # A2
        7.0e4  # Ea2
    ])

    # 2.2 (v17.0) "真实"数据范围 (来自 v16.0 调谐器)
    X_physical_bounds = {
        'C_A0': [0.1, 1.0], 'C_B0': [0.1, 1.0],
        'T': [280.0, 380.0], 't': [60.0, 1000.0]
    }
    X_scaler = {
        'min': np.array([b[0] for b in X_physical_bounds.values()]),
        'max': np.array([b[1] for b in X_physical_bounds.values()])
    }

    print(f"  (v17.0) 真实模型: A+B->P, P->S")
    print(f"  (v17.0) 真实参数: Ea1={w_true_P[3]:.1e}, Ea2={w_true_S[3]:.1e}")
    print(f"  (v17.0) 数据范围: T=[{X_physical_bounds['T'][0]}, {X_physical_bounds['T'][1]}] K")
    print(f"  (v17.0) 数据范围: t=[{X_physical_bounds['t'][0]}, {X_physical_bounds['t'][1]}] s")

    # 2.3 生成 X_norm
    sampler_X = qmc.LatinHypercube(d=4)
    X_norm = sampler_X.random(n=N)

    # 2.4 "反归一化" X
    X_physical = X_scaler['min'] + X_norm * (X_scaler['max'] - X_scaler['min'])

    # 2.5 生成“真实”产率 Y (使用 v14 "无限反应物" 风格)
    Y_obs_physical = np.zeros((N, 1))
    for i in range(N):
        C_A0, C_B0, Ti, t_final_i = X_physical[i, :]
        C_init = [C_A0, C_B0, 0.0, 0.0]
        t_span = [0, t_final_i]
        try:
            sol = solve_ivp(
                fun=_ode_system_DECOMPOSITION,  # (v17.0) 使用新 ODE
                t_span=t_span,
                y0=C_init,
                method=config.ODE_METHOD,
                args=(w_true_P, w_true_S, Ti),  # (v17.0) 使用新参数
                rtol=config.ODE_RTOL,  # (v17.0) 使用高精度
                atol=config.ODE_ATOL
            )
            if sol.success:
                Y_obs_physical[i, 0] = sol.y[2, -1]  # (产物 P)
            else:
                Y_obs_physical[i, 0] = 0.0
        except Exception:
            Y_obs_physical[i, 0] = 0.0

    # 2.6 添加“测量噪声”
    noise_std = 0.02
    epsilon = np.random.normal(0, noise_std, (N, 1))
    Y_obs_physical = Y_obs_physical + epsilon
    Y_obs_physical = np.maximum(Y_obs_physical, 0)

    # 2.7 归一化 Y
    Y_mean = np.mean(Y_obs_physical)
    Y_std = np.std(Y_obs_physical)
    if Y_std < 1e-6: Y_std = 1e-6
    Y_scaler = {'mean': Y_mean, 'std': Y_std}
    Y_norm = (Y_obs_physical - Y_mean) / Y_std

    print(f"  物理产率 (Y_obs) 均值: {Y_mean:.2f} M, 标准差: {Y_std:.2f} M")
    print("------------------------\n")

    # (v17.0) 返回 w_true_P (主反应) 用于最终比较
    return X_norm, Y_norm, w_true_P, X_scaler, Y_scaler, Y_obs_physical