# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import qmc
from scipy.optimize import minimize
import GPy
import warnings

# 导入项目模块
import config_benchmark as cfg
from benchmark_env import EpoxyBenchmarkEnv
from epoxy_reaction import EpoxyReactionODE
from cigp_model_normalized import CIGP_Normalized
from acquisition_functions import AcquisitionFactory

# 压制 GPy 的 F-order 警告 (虽然我们把打印移到了最后，但压制一下更清爽)
warnings.filterwarnings("ignore")


# --- 辅助函数 ---

def normalize(X, bounds):
    """物理值 -> [0, 1]"""
    return (X - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])


def inverse_normalize(X_norm, bounds):
    """[0, 1] -> 物理值"""
    return bounds[:, 0] + X_norm * (bounds[:, 1] - bounds[:, 0])


def optimize_acquisition(model, acq_func, bounds_norm):
    """
    优化采集函数寻找下一个点 (X_next)
    """
    dim = bounds_norm.shape[0]

    # 1. 粗搜索
    n_candidates = 2000
    X_cand = np.random.uniform(0, 1, (n_candidates, dim))
    scores = acq_func.compute(model, X_cand)
    best_idx = np.argmax(scores)
    x0 = X_cand[best_idx]

    # 2. 精修
    def obj(x):
        x = x.reshape(1, -1)
        s = acq_func.compute(model, x)
        return -s[0]

    res = minimize(obj, x0, bounds=[(0, 1)] * dim, method='L-BFGS-B')
    return res.x.reshape(1, -1)


# --- 主流程 ---

def run_benchmark():
    print("==========================================================")
    print(f"   CIGP Benchmark: Epoxy Synthesis (Styrene Oxide)")
    print(f"   Strategy: {cfg.ACQ_FUNC_TYPE} | Total Iters: {cfg.N_BO_ITERS}")
    print("==========================================================\n")

    env = EpoxyBenchmarkEnv()
    bounds_phys = cfg.DESIGN_SPACE['bounds']
    X_scaler = {'min': bounds_phys[:, 0], 'max': bounds_phys[:, 1]}

    # --- 1. LHS 初始化采样 ---
    print(f">>> Phase 1: LHS Sampling ({cfg.N_LHS_SAMPLES} points)...")
    sampler = qmc.LatinHypercube(d=4, seed=42)
    X_init_norm = sampler.random(n=cfg.N_LHS_SAMPLES)
    X_init_phys = inverse_normalize(X_init_norm, bounds_phys)

    X_history = []
    Y_history = []

    # 用于存储打印日志的列表
    iteration_logs = []

    for i in range(cfg.N_LHS_SAMPLES):
        y_obs, y_true = env.run_experiment(X_init_phys[i])
        X_history.append(X_init_phys[i])
        Y_history.append(y_obs)

        # 记录 LHS 阶段的数据
        c_sty, c_paa, temp, time_s = X_init_phys[i]
        log_entry = {
            'iter': f"LHS-{i + 1}",
            'inputs': (c_sty, c_paa, temp, time_s),
            'pred': np.nan,  # LHS阶段无预测
            'obs': y_obs,
            'true': y_true,
            'best': np.max(Y_history)
        }
        iteration_logs.append(log_entry)
        print(f"    Sampled LHS point {i + 1}/{cfg.N_LHS_SAMPLES}...")

    X_train = np.array(X_history)
    Y_train = np.array(Y_history).reshape(-1, 1)

    # --- 2. 贝叶斯优化循环 ---
    print(f"\n>>> Phase 2: Bayesian Optimization ({cfg.N_BO_ITERS} rounds)...")
    print("    (Running optimization quietly to avoid log clutter...)")

    acq_func = AcquisitionFactory.get(cfg.ACQ_FUNC_TYPE, **cfg.ACQ_PARAMS)

    # 定义 model 变量，确保循环结束后依然可以访问它
    model = None

    for iteration in range(cfg.N_BO_ITERS):
        # 2.1 数据预处理
        Y_mean, Y_std = np.mean(Y_train), np.std(Y_train)
        if Y_std < 1e-6: Y_std = 1e-6
        Y_scaler = {'mean': Y_mean, 'std': Y_std}

        X_train_norm = np.asfortranarray(normalize(X_train, bounds_phys))
        Y_train_norm = np.asfortranarray((Y_train - Y_mean) / Y_std)

        # 2.2 构建并训练模型
        physics_model = EpoxyReactionODE(X_scaler=X_scaler)
        kernel = GPy.kern.RBF(input_dim=4, ARD=True)
        model = CIGP_Normalized(X_train_norm, Y_train_norm, physics_model, Y_scaler, kernel=kernel)

        # 约束参数
        model.likelihood.variance.constrain_bounded(*cfg.GP_CONST['noise_var'])
        model.kernel.variance.constrain_bounded(*cfg.GP_CONST['kernel_var'])
        model.kernel.lengthscale.constrain_positive(cfg.GP_CONST['kernel_len'])

        for name, bound in cfg.PHYSICS_BOUNDS.items():
            idx = physics_model.param_names.index(name)
            model.W[idx:idx + 1].constrain_bounded(*bound)

        # 优化 (不打印信息)
        model.optimize(cfg.OPTIMIZER, max_iters=cfg.MAX_ITERS, messages=False)

        # 2.3 采集下一个点
        X_next_norm = optimize_acquisition(model, acq_func, bounds_phys)
        X_next_phys = inverse_normalize(X_next_norm, bounds_phys)

        # 预测 (用于对比)
        mu_norm_pred, _ = model.predict(X_next_norm)
        y_pred = mu_norm_pred[0, 0] * Y_scaler['std'] + Y_scaler['mean']

        # 2.4 实验
        y_new, y_true_val = env.run_experiment(X_next_phys[0])

        # 2.5 更新数据
        X_train = np.vstack([X_train, X_next_phys])
        Y_train = np.vstack([Y_train, [[y_new]]])

        # 2.6 记录日志 (暂不打印)
        c_sty, c_paa, temp, time_s = X_next_phys[0]
        log_entry = {
            'iter': f"BO-{iteration + 1}",
            'inputs': (c_sty, c_paa, temp, time_s),
            'pred': y_pred,
            'obs': y_new,
            'true': y_true_val,
            'best': np.max(Y_train)
        }
        iteration_logs.append(log_entry)

        # 简单的进度提示
        if (iteration + 1) % 5 == 0:
            print(f"    ... Completed {iteration + 1} iterations.")

    # --- 3. 最终汇总报告 ---
    print("\n" + "=" * 95)
    print(f"{'FINAL OPTIMIZATION REPORT':^95}")
    print("=" * 95)
    print(
        f"{'Iter':<8} | {'Styrene':<7} {'PAA':<7} {'Temp(K)':<7} {'Time(s)':<7} | {'Pred':<8} {'Observed':<8} {'True':<8} | {'Best':<8}")
    print("-" * 95)

    for log in iteration_logs:
        c_sty, c_paa, temp, time_s = log['inputs']
        # 处理 LHS 阶段没有预测值的情况
        pred_str = f"{log['pred']:.4f}" if not np.isnan(log['pred']) else "N/A"

        print(f"{log['iter']:<8} | {c_sty:<7.2f} {c_paa:<7.2f} {temp:<7.1f} {time_s:<7.0f} | "
              f"{pred_str:<8} {log['obs']:<8.4f} {log['true']:<8.4f} | {log['best']:<8.4f}")
    print("=" * 95)

    # --- 4. 打印拟合的动力学方程 ---
    print("\n>>> Final Fitted Kinetic Model (Physical Interpretability):")

    if model is not None:
        # 获取最终参数值
        final_w = model.W.values
        names = model.physics_model.param_names

        print("-" * 60)
        print(f"{'Parameter':<15} | {'Fitted Value':<15} | {'Linear Value'}")
        print("-" * 60)

        # 提取关键参数用于打印方程
        fitted_params = {}

        for name, val in zip(names, final_w):
            fitted_params[name] = val

            # 如果是 log 参数，计算线性值方便阅读
            linear_val_str = ""
            if 'log10' in name:
                linear_val = 10 ** val
                linear_val_str = f"{linear_val:.2e}"
            elif 'alpha' in name or 'beta' in name:
                linear_val_str = f"{val:.4f}"

            print(f"{name:<15} | {val:<15.4f} | {linear_val_str}")

        print("-" * 60)

        # 打印方程形式
        logA1, logE1 = fitted_params['log10_A_1'], fitted_params['log10_Ea_1']
        logA2, logE2 = fitted_params['log10_A_2'], fitted_params['log10_Ea_2']
        a1, b1 = fitted_params['alpha_1'], fitted_params['beta_1']
        a2, b2 = fitted_params['alpha_2'], fitted_params['beta_2']

        print("\n[Inferred Reaction Mechanism]")
        print(
            f"Main Reaction (A+B->P): k1 = {10 ** logA1:.1e} * exp(-{10 ** logE1:.0f}/RT) * [A]^{a1:.2f} [B]^{b1:.2f}")
        print(
            f"Side Reaction (P+D->S): k2 = {10 ** logA2:.1e} * exp(-{10 ** logE2:.0f}/RT) * [P]^{a2:.2f} [D]^{b2:.2f}")

        # 与真值对比 (Hardcoded for benchmark context)
        print("\n[Ground Truth Comparison]")
        print(f"Main Ea: True=55.0 kJ/mol vs Fitted={10 ** logE1 / 1000:.1f} kJ/mol")
        print(f"Side Ea: True=85.0 kJ/mol vs Fitted={10 ** logE2 / 1000:.1f} kJ/mol")

    else:
        print("Model was not trained (N_BO_ITERS might be 0).")

    # --- 5. 结果可视化 ---
    print(f"\n>>> Phase 3: Visualization ({cfg.VIS_SAVE_PATH})...")
    plot_results(X_train, Y_train, cfg.N_LHS_SAMPLES)


def plot_results(X, Y, n_lhs):
    """3D 可视化保持不变"""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    T_vals = X[:, 2] - 273.15
    t_vals = X[:, 3]
    ratios = X[:, 1] / (X[:, 0] + 1e-9)
    yields = Y.flatten()

    p1 = ax.scatter(T_vals[:n_lhs], t_vals[:n_lhs], ratios[:n_lhs],
                    c=yields[:n_lhs], cmap='viridis', marker='o', s=40,
                    edgecolor='k', alpha=0.6, label='LHS Init')

    p2 = ax.scatter(T_vals[n_lhs:], t_vals[n_lhs:], ratios[n_lhs:],
                    c=yields[n_lhs:], cmap='viridis', marker='*', s=150,
                    edgecolor='k', label='BO Search', vmin=np.min(yields), vmax=np.max(yields))

    total_points = len(yields)
    for i in range(total_points):
        if i >= n_lhs:
            label = f"{i - n_lhs + 1}"
            ax.text(T_vals[i], t_vals[i], ratios[i], label, fontsize=9, color='red')

    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Time (s)')
    ax.set_zlabel('Equivalence Ratio (PAA/Styrene)')
    ax.set_title('CIGP Benchmark: Epoxy Yield Optimization trajectory')

    cbar = fig.colorbar(p2, ax=ax, shrink=0.6)
    cbar.set_label('Yield (M)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(cfg.VIS_SAVE_PATH, dpi=300)
    plt.show()


if __name__ == "__main__":
    run_benchmark()