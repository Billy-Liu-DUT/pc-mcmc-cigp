# -*- coding: utf-8 -*-
# 1. CPU 锁 (必须放在最前面，防止 Numpy 多线程与 Multiprocessing 冲突)
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import warnings

warnings.simplefilter("ignore")

import numpy as np
import GPy
import time
import multiprocessing
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import qmc

import config

# ==========================================
# 1. 导入算子库 (Model Zoo)
# ==========================================
from simple_reaction import SimpleReactionODE
from reversible_reaction import ReversibleReactionODE
from parallel_reaction import ParallelReactionODE
from series_reaction import SeriesReactionODE
from autocatalytic_reaction import AutocatalyticODE

from cigp_model_normalized import CIGP_Normalized
from simulation_utils import generate_synthetic_data

# 导入通用优化任务函数
from optimization_task import get_lhs_starting_points, run_optimization_task
from plotting_utils import plot_g_err_diagnostic, plot_lhs_diagnostics


# --- 包装函数 (避免 Windows/Mac 多进程 Pickle 错误) ---
def task_wrapper(args):
    return run_optimization_task(*args)


if __name__ == "__main__":
    multiprocessing.freeze_support()

    print(f"\n=======================================================")
    print(f"   CIGP Physics-Informed Training (v28.0 Model Zoo)")
    print(f"=======================================================")
    print(f"当前选用的算子 (Model): {config.PHYSICS_MODEL_TYPE}")
    print(f"待优化的物理参数: {config.CURRENT_PARAM_NAMES}")
    print(f"=======================================================\n")

    # ----------------------------------------------------------------
    # 1. 生成/加载数据
    # ----------------------------------------------------------------
    # 注意: 数据通常由 simulation_utils.py 里的 "真实" 模型生成 (连串反应 A->P->S)
    # 无论我们选什么算子，都是去拟合这份数据，从而观察 g_err 如何修补偏差
    N_data = config.N_DATA
    X_norm, Y_norm, w_true_P, X_scaler, Y_scaler, Y_obs_physical = generate_synthetic_data(N=N_data)

    # 强制数据内存连续 (解决 F-order 警告)
    X_norm = np.asfortranarray(X_norm)
    Y_norm = np.asfortranarray(Y_norm)

    # ----------------------------------------------------------------
    # 2. LHS 采样 (自动适配 config 维度)
    # ----------------------------------------------------------------
    # 维度 = 物理参数数量 + GP参数数量(6)
    start_points = get_lhs_starting_points(config.N_RESTARTS, config.SAMPLING_BOUNDS)

    # ----------------------------------------------------------------
    # 3. 并行训练
    # ----------------------------------------------------------------
    if config.NUM_CORES == 'auto':
        num_cores = max(1, multiprocessing.cpu_count() - 1)
    else:
        num_cores = int(config.NUM_CORES)

    print(f"启动并行池 (Cores: {num_cores})...")

    # 准备任务参数
    tasks_args = []
    for i in range(config.N_RESTARTS):
        tasks_args.append(
            (i, start_points[i], X_norm, Y_norm, config.W_BOUNDS_LOG, X_scaler, Y_scaler)
        )

    t0 = time.time()
    pool = multiprocessing.Pool(processes=num_cores)

    # 使用 tqdm 显示进度条
    results = list(tqdm(pool.imap(task_wrapper, tasks_args),
                        total=config.N_RESTARTS,
                        unit="task",
                        desc="Training CIGP"))

    pool.close()
    pool.join()

    print(f"\n训练完成 (耗时: {time.time() - t0:.2f}s)")

    # ----------------------------------------------------------------
    # 4. 结果分析与最优模型重建
    # ----------------------------------------------------------------
    valid_results = [r for r in results if r[0] != np.inf]
    if not valid_results:
        print("【Error】所有优化任务均失败，请检查参数边界或 ODE 求解器设置。")
        exit()

    best_nll, best_w, best_params = min(valid_results, key=lambda item: item[0])
    print(f"\n--- Global Best Result ---")
    print(f"Min Loss (NLL): {best_nll:.4f}")

    # === 模型工厂: 根据 config 重建最优模型 ===
    model_type = config.PHYSICS_MODEL_TYPE
    n_params = len(config.CURRENT_PARAM_NAMES)
    dummy_init = np.zeros(n_params)  # 占位符，数值稍后被覆盖

    if model_type == 'SIMPLE':
        best_model_physics = SimpleReactionODE(
            W_init=dummy_init, X_scaler=X_scaler, stoichiometry=[1., 1., 1.]
        )

    elif 'REVERSIBLE' in model_type:
        is_var = ('VAR' in model_type)
        best_model_physics = ReversibleReactionODE(
            W_init=dummy_init, X_scaler=X_scaler, variable_orders=is_var
        )

    elif 'PARALLEL' in model_type:
        best_model_physics = ParallelReactionODE(
            W_init=dummy_init, X_scaler=X_scaler, variable_orders=True
        )

    elif 'SERIES' in model_type:
        best_model_physics = SeriesReactionODE(
            W_init=dummy_init, X_scaler=X_scaler
        )

    elif 'AUTOCAT' in model_type:
        best_model_physics = AutocatalyticODE(
            W_init=dummy_init, X_scaler=X_scaler
        )
    else:
        raise ValueError(f"未知的模型类型: {model_type}")

    # 构建 GP 并加载参数
    best_model_kernel = GPy.kern.RBF(input_dim=X_norm.shape[1], ARD=True)

    best_model = CIGP_Normalized(
        X_norm, Y_norm,
        physics_model=best_model_physics,
        Y_scaler=Y_scaler,
        kernel=best_model_kernel,
        lambda_penalty=0.0  # 预测时不需要惩罚
    )

    # 核心：将优化得到的 best_params 填入模型
    best_model.update_model(False)
    best_model.initialize_parameter()
    best_model.param_array[:] = best_params
    best_model.update_model(True)

    # ----------------------------------------------------------------
    # 5. 打印结果
    # ----------------------------------------------------------------
    print("\n--- Found Physics Parameters ---")

    phys_vals = best_model.W.values
    names = config.CURRENT_PARAM_NAMES

    # 智能打印：自动转换 log10 参数
    for i, name in enumerate(names):
        val = phys_vals[i]
        if 'log10' in name:
            linear_val = 10 ** val
            print(f"{name:<20}: {val:>8.4f}  (Linear: {linear_val:.2e})")
        else:
            print(f"{name:<20}: {val:>8.4f}")

    # 打印 GP 误差方差
    # param_array 结构: [Phys..., k_var, l1..l4, n_var]
    gp_idx_start = len(names)
    sigma_k_sq = best_params[gp_idx_start]
    print(f"{'Sigma_k^2 (Model Error)':<20}: {sigma_k_sq:.4f}")

    # ----------------------------------------------------------------
    # 6. 可视化
    # ----------------------------------------------------------------
    if config.VIS_PLOT_DIAGNOSTIC:
        try:
            print("\n正在生成诊断图...")
            plot_lhs_diagnostics(valid_results, best_params)
            plot_g_err_diagnostic(best_model, X_scaler, Y_scaler, X_norm, Y_obs_physical)
            print("绘图完成。")
        except Exception as e:
            print(f"【Warning】绘图失败: {e}")
            import traceback

            traceback.print_exc()