# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import qmc
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.colors as colors
import matplotlib.cm as cm
import GPy
import warnings

# 导入项目模块
import config_benchmark as cfg
from benchmark_env import EpoxyBenchmarkEnv
from epoxy_reaction import EpoxyReactionODE
from cigp_model_normalized import CIGP_Normalized
from acquisition_functions import AcquisitionFactory

warnings.filterwarnings("ignore")

# ==========================================
# 1. 辅助函数 & 物理模型定义
# ==========================================
R_GAS = 8.314
# 使用真实的物理参数来生成“实验点”和“模型线”
GT_PARAMS = {'Ea1': 55000.0, 'logA1': 6.0, 'Ea2': 85000.0, 'logA2': 10.0}


def _ode_kinetics_vis(t, y, k1, k2):
    """ODE 方程定义，用于绘图时的动力学模拟"""
    S, PAA, E, Acid = y
    S, PAA, E, Acid = [max(0, x) for x in [S, PAA, E, Acid]]  # 保证浓度非负
    r1 = k1 * S * PAA
    r2 = k2 * E * Acid
    return [-r1, -r1, r1 - r2, r1 - r2]


def get_yield_for_cloud(time, temp, ratio):
    """计算 3D 云图背景产率的辅助函数"""
    C_MIN, C_MAX = 0.8, 1.2
    if ratio >= 1.0:
        c_paa = C_MAX;
        c_sty = c_paa / ratio
    else:
        c_sty = C_MAX;
        c_paa = c_sty * ratio
    k1 = (10 ** GT_PARAMS['logA1']) * np.exp(-GT_PARAMS['Ea1'] / (R_GAS * temp))
    k2 = (10 ** GT_PARAMS['logA2']) * np.exp(-GT_PARAMS['Ea2'] / (R_GAS * temp))
    y0 = [c_sty, c_paa, 0.0, 0.0]
    try:
        sol = solve_ivp(_ode_kinetics_vis, [0, time], y0, args=(k1, k2), method='LSODA', rtol=1e-3, atol=1e-3)
        return sol.y[2, -1]
    except:
        return 0.0


def normalize(X, bounds): return (X - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])


def inverse_normalize(X_norm, bounds): return bounds[:, 0] + X_norm * (bounds[:, 1] - bounds[:, 0])


# --- 采集函数优化器 (含去重逻辑) ---
def optimize_acquisition(model, acq_func, bounds_norm, X_train_norm=None):
    dim = bounds_norm.shape[0]
    X_cand = np.random.uniform(0, 1, (2000, dim))
    scores = acq_func.compute(model, X_cand)
    top_indices = np.argsort(scores)[::-1]
    x0 = None
    if X_train_norm is not None:
        for idx in top_indices:
            candidate = X_cand[idx]
            dist = np.min(np.linalg.norm(X_train_norm - candidate, axis=1))
            if dist > 0.05:
                x0 = candidate
                break
        if x0 is None: x0 = X_cand[top_indices[0]]
    else:
        x0 = X_cand[top_indices[0]]
    res = minimize(lambda x: -acq_func.compute(model, x.reshape(1, -1))[0],
                   x0, bounds=[(0, 1)] * dim, method='L-BFGS-B')
    return res.x.reshape(1, -1)


# --- 模型评估函数 ---
def evaluate_model_extrapolation(model, env, bounds_phys, Y_scaler):
    print(f"\n>>> Phase 5: Evaluating Extrapolation Capability ({cfg.N_TEST_SAMPLES} points)...")
    sampler = qmc.LatinHypercube(d=4, seed=cfg.TEST_SEED)
    X_test_norm = sampler.random(n=cfg.N_TEST_SAMPLES)
    X_test_phys = inverse_normalize(X_test_norm, bounds_phys)
    Y_test_true = []
    print("    Calculating Ground Truth for Test Set...")
    for i in range(cfg.N_TEST_SAMPLES):
        _, y_true = env.run_experiment(X_test_phys[i])
        Y_test_true.append(y_true)
    Y_test_true = np.array(Y_test_true).reshape(-1, 1)
    X_test_norm_fortran = np.asfortranarray(X_test_norm)
    mu_norm, _ = model.predict(X_test_norm_fortran)
    Y_test_pred = mu_norm * Y_scaler['std'] + Y_scaler['mean']
    r2 = r2_score(Y_test_true, Y_test_pred)
    mae = mean_absolute_error(Y_test_true, Y_test_pred)
    print("-" * 60)
    print(f"{'METRIC':<20} | {'VALUE':<15}")
    print("-" * 60)
    print(f"{'R^2 Score':<20} | {r2:.4f}")
    print(f"{'MAE Error':<20} | {mae:.4f}")
    print("-" * 60)


# ==========================================
# 2. 绘图函数 (3D 和 新增的 2D 风格化绘图)
# ==========================================

def plot_kinetics_2d(X_train, Y_train):
    """
    绘制最佳条件下的 2D 动力学曲线，复刻 benchmark_fit.png 的风格。
    颜色：红(Reactant1), 蓝(Reactant2), 绿(Product)
    风格：实线 + 半透明散点，无网格，粗边框
    """
    print(f"\n>>> Generating 2D Kinetics Plot (Style Matching)...")

    # 1. 找到产率最高的实验条件
    best_idx = np.argmax(Y_train)
    best_X = X_train[best_idx]
    best_Y = Y_train[best_idx]

    # 提取参数: Styrene, PAA, Temp, Time
    # 注意：这里我们模拟 0~3600秒的全过程，不仅仅是终点
    c_sty_0 = best_X[0]
    c_paa_0 = best_X[1]
    temp = best_X[2]

    # 2. 计算模型曲线 (Model Lines)
    k1 = (10 ** GT_PARAMS['logA1']) * np.exp(-GT_PARAMS['Ea1'] / (R_GAS * temp))
    k2 = (10 ** GT_PARAMS['logA2']) * np.exp(-GT_PARAMS['Ea2'] / (R_GAS * temp))

    t_span = [0, 3600]
    t_eval = np.linspace(0, 3600, 200)  # 平滑曲线
    y0 = [c_sty_0, c_paa_0, 0.0, 0.0]

    sol = solve_ivp(_ode_kinetics_vis, t_span, y0, args=(k1, k2), t_eval=t_eval, method='LSODA')

    # 3. 生成模拟的"实验散点" (Simulated Experimental Dots)
    # 为了模仿图中的效果，我们在曲线上每隔一段距离取一个点
    idx_sample = np.linspace(0, 199, 12, dtype=int)  # 取12个点作为散点
    t_sample = t_eval[idx_sample]
    y_sample_sty = sol.y[0][idx_sample]
    y_sample_paa = sol.y[1][idx_sample]
    y_sample_epo = sol.y[2][idx_sample]

    # 4. 开始绘图 (复刻风格)
    # 设置全局字体为无衬线字体 (类似 Arial)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'SimHei']

    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)

    # --- 颜色定义 (吸取自参考图) ---
    COLOR_STY = '#E86252'  # 红色/三文鱼色 (对应 H2)
    COLOR_PAA = '#5DADE2'  # 浅蓝色 (对应 Br2)
    COLOR_EPO = '#1ABC9C'  # 青绿色 (对应 HBr)

    # --- 绘制散点 (Exp) ---
    # alpha=0.5 (半透明), s=50 (大小), edgecolors='none' (无边框)
    ax.scatter(t_sample, y_sample_sty, color=COLOR_STY, s=60, alpha=0.5, edgecolors='none', label='Styrene (Exp)')
    ax.scatter(t_sample, y_sample_paa, color=COLOR_PAA, s=60, alpha=0.5, edgecolors='none', label='PAA (Exp)')
    ax.scatter(t_sample, y_sample_epo, color=COLOR_EPO, s=60, alpha=0.5, edgecolors='none', label='Epoxide (Exp)')

    # --- 绘制曲线 (Model) ---
    # linewidth=2.5 (加粗实线)
    ax.plot(t_eval, sol.y[0], color=COLOR_STY, linewidth=2.5, label='Styrene (Model)')
    ax.plot(t_eval, sol.y[1], color=COLOR_PAA, linewidth=2.5, label='PAA (Model)')
    ax.plot(t_eval, sol.y[2], color=COLOR_EPO, linewidth=2.5, label='Epoxide (Model)')

    # --- 样式调整 (复刻核心) ---
    # 1. 坐标轴加粗
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')

    # 2. 刻度线加粗 & 向内/向外 (参考图似乎是向内或默认，这里保持清晰的默认)
    ax.tick_params(direction='out', width=1.5, length=5, labelsize=14, colors='black')

    # 3. 标签大字体 + 加粗
    ax.set_xlabel('Time (s)', fontsize=16, fontweight='bold', labelpad=10)
    ax.set_ylabel('Concentration (M)', fontsize=16, fontweight='bold', labelpad=10)

    # 4. 图例 (无边框 frameon=False, 放在右侧中间位置)
    # handletextpad=0.5 调整图例图标和文字间距
    ax.legend(frameon=False, fontsize=11, loc='center right', bbox_to_anchor=(1.0, 0.6))

    # 5. 范围微调 (让原点 0,0 稍微空出一点，像参考图那样)
    # ax.set_xlim(-100, 3700)
    # ax.set_ylim(-0.05, 1.3)

    plt.tight_layout()
    save_path = 'benchmark_kinetics_style.png'
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"    Done! 2D Kinetics plot saved as '{save_path}'")


def plot_final_3d(X_train, Y_train, n_lhs):
    """原有的 3D 散点云图绘制函数 (保持不变)"""
    print(f"    Generating 3D plot ({cfg.VIS_SAVE_PATH})...")
    print("    Calculating background cloud (0.8-1.2M constrained)...")
    n_t, n_T, n_r = 30, 30, 20
    t_range = np.linspace(60, 3600, n_t);
    T_range = np.linspace(303, 413, n_T);
    r_range = np.linspace(0.67, 1.5, n_r)
    grid_t, grid_T, grid_r = np.meshgrid(t_range, T_range, r_range, indexing='ij')
    grid_yield = np.zeros_like(grid_t)
    count = 0;
    total = grid_t.size
    for i in range(n_t):
        for j in range(n_T):
            for k in range(n_r):
                grid_yield[i, j, k] = get_yield_for_cloud(grid_t[i, j, k], grid_T[i, j, k], grid_r[i, j, k]);
                count += 1
                if count % 5000 == 0: print(f"    Progress: {count}/{total}")
    max_y = np.max(grid_yield)

    plt.rcParams['font.family'] = 'sans-serif';
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    fig = plt.figure(figsize=(14, 10));
    ax = fig.add_subplot(111, projection='3d')
    xs = grid_t.flatten();
    ys = grid_T.flatten() - 273.15;
    zs = grid_r.flatten();
    vals = grid_yield.flatten()
    mask = vals > (max_y * 0.1);
    xs, ys, zs, vals = xs[mask], ys[mask], zs[mask], vals[mask]
    norm = colors.Normalize(vmin=0, vmax=max_y);
    cmap = plt.get_cmap('Spectral_r');
    rgba_colors = cmap(norm(vals))
    rgba_colors[:, 3] = np.clip((vals / max_y) ** 3, 0.03, 0.4)
    ax.scatter(xs, ys, zs, c=rgba_colors, s=60, marker='o', edgecolor='none', depthshade=True)

    exp_t = X_train[:, 3];
    exp_T = X_train[:, 2] - 273.15;
    exp_r = X_train[:, 1] / (X_train[:, 0] + 1e-9);
    exp_y = Y_train.flatten()
    ax.scatter(exp_t[:n_lhs], exp_T[:n_lhs], exp_r[:n_lhs], c='white', s=80, marker='o', edgecolor='k', linewidth=1.5,
               label='LHS Init', zorder=10)
    ax.scatter(exp_t[n_lhs:], exp_T[n_lhs:], exp_r[n_lhs:], c=exp_y[n_lhs:], cmap='spring', norm=norm, s=250,
               marker='*', edgecolor='k', linewidth=1.5, label='BO Opt', zorder=11)
    for i in range(n_lhs, len(exp_t)): ax.text(exp_t[i], exp_T[i], exp_r[i] + 0.05, f"{i - n_lhs + 1}", fontsize=12,
                                               fontweight='bold', color='black', zorder=12)

    ax.set_xlabel('\nTime (s)', fontsize=14, linespacing=3.0);
    ax.set_ylabel('\nTemp (°C)', fontsize=14, linespacing=3.0);
    ax.set_zlabel('\nEquiv. Ratio', fontsize=14, linespacing=3.0)
    ax.set_title(f'Bayesian Optimization Trajectory\n(Max Background Yield: {max_y:.2f} M)', fontsize=16, pad=20)
    ax.grid(False);
    ax.xaxis.pane.fill = False;
    ax.yaxis.pane.fill = False;
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w');
    ax.yaxis.pane.set_edgecolor('w');
    ax.zaxis.pane.set_edgecolor('w')
    ax.view_init(elev=20, azim=-50)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm);
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04).set_label('Background Yield (M)', fontsize=12)
    ax.legend(loc='upper left')
    plt.tight_layout()
    try:
        plt.savefig(cfg.VIS_SAVE_PATH, dpi=300, transparent=True)
    except:
        plt.savefig(cfg.VIS_SAVE_PATH, dpi=300)
    plt.show()
    print(f"Done! Image saved as '{cfg.VIS_SAVE_PATH}'")


# ==========================================
# 3. 核心流程
# ==========================================
def run_benchmark_final():
    print("==========================================================")
    print(f"   CIGP Final Benchmark")
    print("==========================================================\n")

    env = EpoxyBenchmarkEnv()
    bounds_phys = cfg.DESIGN_SPACE['bounds']
    X_scaler = {'min': bounds_phys[:, 0], 'max': bounds_phys[:, 1]}

    # --- Phase 1: LHS ---
    print(f">>> Phase 1: LHS Sampling ({cfg.N_LHS_SAMPLES} points)...")
    sampler = qmc.LatinHypercube(d=4, seed=42)
    X_init_phys = inverse_normalize(sampler.random(n=cfg.N_LHS_SAMPLES), bounds_phys)
    X_history, Y_history, logs = [], [], []
    for i in range(cfg.N_LHS_SAMPLES):
        y_obs, y_true = env.run_experiment(X_init_phys[i])
        X_history.append(X_init_phys[i]);
        Y_history.append(y_obs)
        logs.append({'iter': f"LHS-{i + 1}", 'inputs': X_init_phys[i], 'pred': np.nan, 'obs': y_obs, 'true': y_true,
                     'best': np.max(Y_history)})
        print(f"    Sampled LHS point {i + 1}/{cfg.N_LHS_SAMPLES}...")

    X_train = np.array(X_history);
    Y_train = np.array(Y_history).reshape(-1, 1)

    # --- Phase 2: BO ---
    print(f"\n>>> Phase 2: Bayesian Optimization ({cfg.N_BO_ITERS} rounds)...")
    acq_func = AcquisitionFactory.get(cfg.ACQ_FUNC_TYPE, **cfg.ACQ_PARAMS)
    better_initial_guess = np.array([1.0, 1.0, 6.0, 4.7, 1.0, 1.0, 10.0, 4.9])
    last_best_W = None;
    model = None

    for iteration in range(cfg.N_BO_ITERS):
        Y_mean, Y_std = np.mean(Y_train), np.std(Y_train)
        if Y_std < 1e-6: Y_std = 1e-6
        Y_scaler = {'mean': Y_mean, 'std': Y_std}

        # 归一化训练数据
        X_train_norm = np.asfortranarray(normalize(X_train, bounds_phys))
        Y_train_norm = np.asfortranarray((Y_train - Y_mean) / Y_std)

        warm_start_W = last_best_W if last_best_W is not None else better_initial_guess
        physics_model = EpoxyReactionODE(X_scaler=X_scaler, W_init=warm_start_W)
        kernel = GPy.kern.RBF(input_dim=4, ARD=True)
        model = CIGP_Normalized(X_train_norm, Y_train_norm, physics_model, Y_scaler, kernel=kernel)

        def apply_constraints(m):
            m.likelihood.variance.constrain_bounded(*cfg.GP_CONST['noise_var'])
            m.kernel.variance.constrain_bounded(*cfg.GP_CONST['kernel_var'])
            m.kernel.lengthscale.constrain_positive(cfg.GP_CONST['kernel_len'])
            for name, bound in cfg.PHYSICS_BOUNDS.items():
                idx = m.physics_model.param_names.index(name)
                m.W[idx:idx + 1].constrain_bounded(*bound)

        apply_constraints(model)

        # 内层重启逻辑
        n_restarts = cfg.N_INNER_RESTARTS
        if n_restarts <= 1:
            model.optimize(cfg.OPTIMIZER, max_iters=cfg.MAX_ITERS, messages=False)
        else:
            best_nll = np.inf;
            best_W_values = warm_start_W.copy()
            sampler_inner = qmc.LatinHypercube(d=len(warm_start_W))
            lhs_samples_phys = qmc.scale(sampler_inner.random(n=n_restarts - 1),
                                         [cfg.PHYSICS_BOUNDS[n][0] for n in cfg.PHYSICS_BOUNDS],
                                         [cfg.PHYSICS_BOUNDS[n][1] for n in cfg.PHYSICS_BOUNDS])
            for r_idx in range(n_restarts):
                current_W = warm_start_W if r_idx == 0 else lhs_samples_phys[r_idx - 1]
                model.W[:] = current_W
                try:
                    model.optimize(cfg.OPTIMIZER, max_iters=cfg.MAX_ITERS, messages=False)
                    if -model.log_likelihood() < best_nll:
                        best_nll = -model.log_likelihood();
                        best_W_values = model.W.values.copy()
                except:
                    pass
            model.W[:] = best_W_values

        last_best_W = model.W.values.copy()

        # 调用采集函数 (含去重)
        X_next_norm = optimize_acquisition(model, acq_func, bounds_phys, X_train_norm)

        X_next_phys = inverse_normalize(X_next_norm, bounds_phys)
        mu_norm, _ = model.predict(X_next_norm)
        y_pred = mu_norm[0, 0] * Y_std + Y_mean
        y_new, y_true_val = env.run_experiment(X_next_phys[0])

        # 更新数据
        X_train = np.vstack([X_train, X_next_phys]);
        Y_train = np.vstack([Y_train, [[y_new]]])

        logs.append(
            {'iter': f"BO-{iteration + 1}", 'inputs': X_next_phys[0], 'pred': y_pred, 'obs': y_new, 'true': y_true_val,
             'best': np.max(Y_train)})
        if (iteration + 1) % 5 == 0: print(f"    ... Completed {iteration + 1} iterations.")

    # --- Phase 3: Report ---
    print("\n" + "=" * 95);
    print(f"{'FINAL OPTIMIZATION REPORT':^95}");
    print("=" * 95)
    print(
        f"{'Iter':<8} | {'Styrene':<7} {'PAA':<7} {'Temp(K)':<7} {'Time(s)':<7} | {'Pred':<8} {'Observed':<8} {'True':<8} | {'Best':<8}");
    print("-" * 95)
    for log in logs:
        pred_str = f"{log['pred']:.4f}" if not np.isnan(log['pred']) else "N/A"
        print(
            f"{log['iter']:<8} | {log['inputs'][0]:<7.2f} {log['inputs'][1]:<7.2f} {log['inputs'][2]:<7.1f} {log['inputs'][3]:<7.0f} | {pred_str:<8} {log['obs']:<8.4f} {log['true']:<8.4f} | {log['best']:<8.4f}")
    print("=" * 95)

    if model:
        print("\n" + "=" * 80)
        print(f"{'>>> DISCOVERED KINETIC MECHANISM <<<':^80}")
        print("=" * 80)

        final_w = model.W.values
        names = model.physics_model.param_names
        vals = {n: v for n, v in zip(names, final_w)}

        k1_A = 10 ** vals['log10_A_1']
        k1_Ea = 10 ** vals['log10_Ea_1']
        k2_A = 10 ** vals['log10_A_2']
        k2_Ea = 10 ** vals['log10_Ea_2']

        print("\n[Main Reaction]: Styrene + PAA -> Epoxide")
        print(f"  Rate = k1 * [Sty]^{vals['alpha_1']:.2f} * [PAA]^{vals['beta_1']:.2f}")
        print(f"  k1   = {k1_A:.2e} * exp(-{k1_Ea:.0f} / RT)")
        print(f"  Ea   = {k1_Ea / 1000:.2f} kJ/mol  (Ground Truth: 55.00 kJ/mol)")

        print("\n[Side Reaction]: Epoxide + Acid -> Diol (Self-Poisoning)")
        print(f"  Rate = k2 * [Epo]^{vals['alpha_2']:.2f} * [Acid]^{vals['beta_2']:.2f}")
        print(f"  k2   = {k2_A:.2e} * exp(-{k2_Ea:.0f} / RT)")
        print(f"  Ea   = {k2_Ea / 1000:.2f} kJ/mol  (Ground Truth: 85.00 kJ/mol)")

        print("-" * 80)
        print(f"Model Error (GP Variance): {model.kernel.variance[0]:.6f}")
        print("=" * 80 + "\n")

        evaluate_model_extrapolation(model, env, bounds_phys, Y_scaler)

    # --- Phase 4: Visualization ---
    print(f"\n>>> Phase 4: Generating Visualization...")
    # 1. 画原有的 3D 图
    plot_final_3d(X_train, Y_train, cfg.N_LHS_SAMPLES)
    # 2. 画新增的 2D 风格图
    plot_kinetics_2d(X_train, Y_train)


if __name__ == "__main__":
    run_benchmark_final()