# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import config

# (v17.0) 循环导入 (Cyclic Import) 依赖
from cigp_model_normalized import CIGP_Normalized
# (v17.0) 导入 *新* 的真实 ODE
from simulation_utils import _ode_system_DECOMPOSITION

"""
==================================================================================
v17.0 更新:
- (ODE) "Y_true" 图现在使用 "产物分解" (v16.0) ODE。
- (Params) "Y_true" 图现在使用 v16.0 调谐的 "真实" 参数。
- (Range) "Y_true" 图现在使用 v16.0 调谐的 "真实" T/t 范围。
==================================================================================
"""


# --- 6. 【v17.0 更新】 可视化函数 (回答浓度问题) ---
def plot_g_err_diagnostic(model: CIGP_Normalized, X_scaler: dict, Y_scaler: dict, X_norm: np.ndarray,
                          Y_physical_obs: np.ndarray):
    try:
        plt.rcParams['font.sans-serif'] = [config.VIS_FONT]
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"【警告】设置中文字体 '{config.VIS_FONT}' 失败: {e}。图表可能出现乱码。")

    print("\n--- 正在生成“循环3：机理诊断” 2D 可视化图 (v17.0 范围)... ---")

    # (v17.0) 匹配 v16.0 的“非凸”数据范围
    T_phys_axis = np.linspace(X_scaler['min'][2], X_scaler['max'][2], 30)  # [280, 380]
    t_phys_axis = np.linspace(X_scaler['min'][3], X_scaler['max'][3], 30)  # [60, 50000]
    T_grid_phys, t_grid_phys = np.meshgrid(T_phys_axis, t_phys_axis)

    # --- (v17.0) 回答用户问题: 浓度 ---
    C_A_norm = 1.0  # (即 C_A0 = 1.0 M, 物理最大值)
    C_B_norm = 1.0  # (即 C_B0 = 1.0 M, 物理最大值)
    print(f"  (图表使用归一化浓度: C_A_norm = {C_A_norm}, C_B_norm = {C_B_norm})")
    # ------------------------------------

    T_norm_grid = (T_grid_phys.flatten() - X_scaler['min'][2]) / (X_scaler['max'][2] - X_scaler['min'][2])
    t_norm_grid = (t_grid_phys.flatten() - X_scaler['min'][3]) / (X_scaler['max'][3] - X_scaler['min'][3])

    X_test_norm = np.vstack([
        np.full_like(T_norm_grid, C_A_norm),
        np.full_like(T_norm_grid, C_B_norm),
        T_norm_grid,
        t_norm_grid
    ]).T

    # (预测 g_err 和 f_phys - 无需更改)
    mu_g_err_norm, _ = model.predict_g_err(X_test_norm)
    mu_g_err_physical = mu_g_err_norm * Y_scaler['std']
    Z_g_err = mu_g_err_physical.reshape(T_grid_phys.shape)

    mu_f_phys_physical = model.physics_model.compute_mean(X_test_norm, model.W.values)
    Z_f_phys = mu_f_phys_physical.reshape(T_grid_phys.shape)

    Z_Y_pred = Z_f_phys + Z_g_err

    # (计算“真实”的 Y_obs - v17.0 更新)
    Y_true_physical = np.zeros((X_test_norm.shape[0], 1))

    # (v17.0) 确保使用 v17.0 的“真实”参数 (来自 v16.0 调谐器)
    w_true_P = np.array([1.0, 1.0, 5e6, 5.5e4])
    w_true_S = np.array([1.0, 0.0, 1e6, 7.0e4])

    X_test_physical = X_scaler['min'] + X_test_norm * (X_scaler['max'] - X_scaler['min'])

    for i in range(X_test_physical.shape[0]):
        C_A0, C_B0, Ti, t_final_i = X_test_physical[i, :]
        C_init = [C_A0, C_B0, 0.0, 0.0]
        t_span = [0, t_final_i]
        sol = solve_ivp(
            fun=_ode_system_DECOMPOSITION,  # (v17.0) 使用新 ODE
            t_span=t_span,
            y0=C_init,
            method=config.ODE_METHOD,
            args=(w_true_P, w_true_S, Ti),  # (v17.0) 使用新参数
            rtol=config.ODE_RTOL,  # (v17.0) 使用高精度
            atol=config.ODE_ATOL
        )
        if sol.success: Y_true_physical[i, 0] = sol.y[2, -1]

    Z_Y_true = Y_true_physical.reshape(T_grid_phys.shape)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    (ax1, ax2), (ax3, ax4) = axes

    def plot_heatmap(ax, X, Y, Z, title, data_points=None, vmin=None, vmax=None):
        if vmin is None:
            vmin = np.min(Z)
            vmax = np.max(Z)
        c = ax.contourf(X, Y, Z, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel('温度 (K)')
        ax.set_ylabel('时间 (s)')
        fig.colorbar(c, ax=ax)
        if data_points is not None:
            ax.plot(data_points[:, 2], data_points[:, 3], 'kx', markersize=5, alpha=0.5)

    vmin = min(np.min(Z_f_phys), np.min(Z_Y_pred), np.min(Z_Y_true))
    vmax = max(np.max(Z_f_phys), np.max(Z_Y_pred), np.max(Z_Y_true))

    X_physical = X_scaler['min'] + X_norm * (X_scaler['max'] - X_scaler['min'])

    plot_heatmap(ax1, T_grid_phys, t_grid_phys, Z_Y_true, '“真实”产率 (Y_true)', data_points=X_physical, vmin=vmin,
                 vmax=vmax)
    plot_heatmap(ax2, T_grid_phys, t_grid_phys, Z_f_phys, '$f_{phys}$ (物理模型) 预测', vmin=vmin, vmax=vmax)
    plot_heatmap(ax3, T_grid_phys, t_grid_phys, Z_g_err, '$g_{err}$ (模型发现的“差距”)')
    plot_heatmap(ax4, T_grid_phys, t_grid_phys, Z_Y_pred, '$Y_{pred} = f_{phys} + g_{err}$ (最终拟合)',
                 data_points=X_physical, vmin=vmin, vmax=vmax)

    plt.tight_layout()
    plt.show()


# --- 7. LHS 运行结果可视化 (v9.0, 保持不变) ---
def plot_lhs_diagnostics(all_results: list, best_param_array: np.ndarray):
    """
    可视化 N_RESTARTS 次运行的 NLL vs sigma_k^2。
    """
    print("--- 正在生成“v9.0 LHS 诊断”可视化图... ---")

    try:
        plt.rcParams['font.sans-serif'] = [config.VIS_FONT]
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"【警告】设置中文字体 '{config.VIS_FONT}' 失败: {e}。图表可能出现乱码。")

    n_restarts = len(all_results)
    nlls = np.zeros(n_restarts)
    sigma_k_sq_vals = np.zeros(n_restarts)
    sigma_n_sq_vals = np.zeros(n_restarts)

    IDX_K_VAR = 4
    IDX_N_VAR = 9

    best_nll = np.inf
    best_idx = -1

    for i, res in enumerate(all_results):
        nll, _, param_array = res
        nlls[i] = nll

        if param_array is not None:
            sigma_k_sq_vals[i] = param_array[IDX_K_VAR]
            sigma_n_sq_vals[i] = param_array[IDX_N_VAR]

            if nll < best_nll:
                best_nll = nll
                best_idx = i
        else:
            sigma_k_sq_vals[i] = np.nan
            sigma_n_sq_vals[i] = np.nan

    # 打印日志
    print("\n--- (v9.0) LHS 运行摘要 (前 10 次): ---")
    print(f"| {'ID':>2} | {'NLL (Loss)':>10} | {'sigma_k^2':>10} | {'sigma_n^2':>10} |")
    print("|" + "-" * 4 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 12 + "|")

    sorted_indices = np.argsort(nlls)
    for i in range(min(n_restarts, 10)):
        idx = sorted_indices[i]
        print(f"| {idx + 1:>2} | {nlls[idx]:>10.2f} | {sigma_k_sq_vals[idx]:>10.4f} | {sigma_n_sq_vals[idx]:>10.6f} |")
    print("...")

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))

    sc = ax.scatter(sigma_k_sq_vals, nlls, c=sigma_n_sq_vals,
                    cmap='plasma', alpha=0.7,
                    label='LHS 运行 (颜色=sigma_n^2)')

    if best_idx != -1:
        ax.scatter(sigma_k_sq_vals[best_idx], nlls[best_idx],
                   c='red', s=100, marker='*',
                   edgecolor='black', label=f'全局最优 (ID: {best_idx + 1})')

    ax.set_xlabel('$\sigma_k^2$ (结构性误差方差)')
    ax.set_ylabel('NLL (Loss)')
    ax.set_title('v9.0 LHS 诊断: NLL vs. $\sigma_k^2$')
    ax.legend()
    fig.colorbar(sc, label='$\sigma_n^2$ (测量噪声方差)')
    plt.tight_layout()
    plt.show()