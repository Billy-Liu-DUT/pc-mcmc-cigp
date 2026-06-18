# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import qmc, norm
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.colors as colors
import matplotlib.cm as cm
import GPy
import warnings

import config_benchmark as cfg
from benchmark_env import EpoxyBenchmarkEnv

warnings.filterwarnings("ignore")

# ==========================================
# 1. 辅助函数
# ==========================================
R_GAS = 8.314
GT_PARAMS = {'Ea1': 55000.0, 'logA1': 6.0, 'Ea2': 85000.0, 'logA2': 10.0}

# --- 【新增】 定义固定的核参数 (请根据您的CIGP实际值调整) ---
FIXED_LENGTHSCALE = 0.5  # 固定长度尺度 (针对归一化数据，0.3~0.8 通常比较合理)
FIXED_VARIANCE = 1.0  # 固定核方差 (归一化Y后通常为1.0)
FIXED_NOISE = 1e-4  # 固定观测噪声 (假设噪声很小)


def _ode_kinetics_vis(t, y, k1, k2):
    S, PAA, E, Acid = y
    S, PAA, E, Acid = [max(0, x) for x in [S, PAA, E, Acid]]
    r1 = k1 * S * PAA;
    r2 = k2 * E * Acid
    return [-r1, -r1, r1 - r2, r1 - r2]


def get_yield_for_cloud(time, temp, ratio):
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


class StandardBOModelWrapper:
    def __init__(self, model, Y_mean, Y_std):
        self.model = model
        self.X = model.X
        self.Y_mean = Y_mean
        self.Y_std = Y_std
        self.W = None;
        self.physics_model = None

    def predict(self, X_new): return self.model.predict(X_new)

    def predict_g_err(self, X_new): return self.model.predict(X_new)


# --- 本地标准 EI 采集函数 ---
class StandardEIAcquisition:
    def __init__(self, xi=0.01):
        self.xi = xi

    def compute(self, model_wrapper, X):
        mu, var = model_wrapper.predict(X)
        sigma = np.sqrt(var)
        Y_train = model_wrapper.model.Y
        y_best = np.max(Y_train)

        with np.errstate(divide='warn'):
            imp = mu - y_best - self.xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma < 1e-9] = 0.0
        return ei


def optimize_acquisition(model_wrapper, acq_func, bounds_norm):
    dim = bounds_norm.shape[0]
    X_cand = np.random.uniform(0, 1, (2000, dim))
    scores = acq_func.compute(model_wrapper, X_cand)
    x0 = X_cand[np.argmax(scores)]
    res = minimize(lambda x: -acq_func.compute(model_wrapper, x.reshape(1, -1))[0], x0, bounds=[(0, 1)] * dim,
                   method='L-BFGS-B')
    return res.x.reshape(1, -1)


# --- 外推评估 ---
def evaluate_model_extrapolation(model, env, bounds_phys, Y_mean, Y_std):
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
    Y_test_pred = mu_norm * Y_std + Y_mean

    r2 = r2_score(Y_test_true, Y_test_pred)
    mae = mean_absolute_error(Y_test_true, Y_test_pred)

    print("-" * 60)
    print(f"{'METRIC':<20} | {'VALUE':<15}")
    print("-" * 60)
    print(f"{'R^2 Score':<20} | {r2:.4f}")
    print(f"{'MAE Error':<20} | {mae:.4f}")
    print("-" * 60)


# ==========================================
# 2. 核心流程
# ==========================================
def run_standard_bo():
    print("==========================================================")
    print(f"   Standard BO Benchmark (Fixed Kernel)")
    print("==========================================================\n")

    env = EpoxyBenchmarkEnv()
    bounds_phys = cfg.DESIGN_SPACE['bounds']

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

    print(f"\n>>> Phase 2: Standard BO Loop ({cfg.N_BO_ITERS} rounds)...")

    # 使用本地定义的 StandardEI, xi=0.6
    print("    Using Local Standard EI Acquisition (Pure GP, xi=0.6)...")
    acq_func = StandardEIAcquisition(xi=0.6)

    model = None
    for iteration in range(cfg.N_BO_ITERS):
        Y_mean, Y_std = np.mean(Y_train), np.std(Y_train)
        if Y_std < 1e-6: Y_std = 1e-6
        X_train_norm = np.asfortranarray(normalize(X_train, bounds_phys))
        Y_train_norm = np.asfortranarray((Y_train - Y_mean) / Y_std)

        # -------------------------------------------------------------
        # 【关键修改】固定核参数，不进行优化
        # -------------------------------------------------------------
        kernel = GPy.kern.RBF(input_dim=4, ARD=True)
        model = GPy.models.GPRegression(X_train_norm, Y_train_norm, kernel)

        # 1. 设定固定的参数值 (根据开头定义的常量)
        model.rbf.lengthscale[:] = FIXED_LENGTHSCALE
        model.rbf.variance[:] = FIXED_VARIANCE
        model.Gaussian_noise.variance[:] = FIXED_NOISE

        # 2. 锁定参数 (Fix them)
        model.rbf.lengthscale.fix()
        model.rbf.variance.fix()
        model.Gaussian_noise.variance.fix()

        # 3. 禁用优化 (Do NOT call optimize)
        # model.optimize(...)  <-- Commented out

        # -------------------------------------------------------------

        model_wrapper = StandardBOModelWrapper(model, Y_mean, Y_std)
        X_next_norm = optimize_acquisition(model_wrapper, acq_func, bounds_phys)
        X_next_phys = inverse_normalize(X_next_norm, bounds_phys)
        mu_norm, _ = model.predict(X_next_norm)
        y_pred = mu_norm[0, 0] * Y_std + Y_mean
        y_new, y_true_val = env.run_experiment(X_next_phys[0])
        X_train = np.vstack([X_train, X_next_phys]);
        Y_train = np.vstack([Y_train, [[y_new]]])
        logs.append(
            {'iter': f"BO-{iteration + 1}", 'inputs': X_next_phys[0], 'pred': y_pred, 'obs': y_new, 'true': y_true_val,
             'best': np.max(Y_train)})
        if (iteration + 1) % 5 == 0: print(f"    ... Completed {iteration + 1} iterations.")

    # --- Report ---
    print("\n" + "=" * 95);
    print(f"{'FINAL OPTIMIZATION REPORT (STANDARD BO - FIXED KERNEL)':^95}");
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
        Y_mean, Y_std = np.mean(Y_train), np.std(Y_train)
        if Y_std < 1e-6: Y_std = 1e-6
        evaluate_model_extrapolation(model, env, bounds_phys, Y_mean, Y_std)

    vis_filename = 'benchmark_standard_bo_vis.png'
    print(f"\n>>> Phase 4: Generating 3D Visualization ({vis_filename})...")
    plot_final_3d(X_train, Y_train, cfg.N_LHS_SAMPLES, vis_filename)


def plot_final_3d(X_train, Y_train, n_lhs, filename):
    print("    Calculating background cloud (0.8-1.2M constrained)...")
    n_t, n_T, n_r = 30, 30, 20
    t_range = np.linspace(60, 3600, n_t)
    T_range = np.linspace(303, 413, n_T)
    r_range = np.linspace(0.67, 1.5, n_r)
    grid_t, grid_T, grid_r = np.meshgrid(t_range, T_range, r_range, indexing='ij')

    grid_yield = np.zeros_like(grid_t)
    count = 0
    total = grid_t.size
    for i in range(n_t):
        for j in range(n_T):
            for k in range(n_r):
                grid_yield[i, j, k] = get_yield_for_cloud(grid_t[i, j, k], grid_T[i, j, k], grid_r[i, j, k])
                count += 1
    max_y = np.max(grid_yield)

    # Setup Plot
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei', 'DejaVu Sans']
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Data Prep for Cloud
    xs = grid_t.flatten()
    ys = grid_T.flatten() - 273.15
    zs = grid_r.flatten()
    vals = grid_yield.flatten()
    mask = vals > (max_y * 0.1)
    xs, ys, zs, vals = xs[mask], ys[mask], zs[mask], vals[mask]

    # CIGP 风格颜色与透明度
    norm = colors.Normalize(vmin=0, vmax=max_y)
    cmap = plt.get_cmap('Spectral_r')
    rgba_colors = cmap(norm(vals))
    rgba_colors[:, 3] = np.clip((vals / max_y) ** 3, 0.03, 0.4)
    ax.scatter(xs, ys, zs, c=rgba_colors, s=60, marker='o', edgecolor='none', depthshade=True)

    # Plot Experiment Points
    exp_t = X_train[:, 3]
    exp_T = X_train[:, 2] - 273.15
    exp_r = X_train[:, 1] / (X_train[:, 0] + 1e-9)

    # LHS (White Circles)
    ax.scatter(exp_t[:n_lhs], exp_T[:n_lhs], exp_r[:n_lhs],
               c='white', s=80, marker='o', edgecolor='k', linewidth=1.5,
               label='LHS Init', zorder=10)

    # Standard BO (Red Triangles, Solid Color)
    ax.scatter(exp_t[n_lhs:], exp_T[n_lhs:], exp_r[n_lhs:],
               c='red', s=200, marker='^',
               edgecolor='k', linewidth=1.5, label='Standard BO', zorder=11)

    # Labels
    for i in range(n_lhs, len(exp_t)):
        ax.text(exp_t[i], exp_T[i], exp_r[i] + 0.05, f"{i - n_lhs + 1}",
                fontsize=10, fontweight='bold', color='black', zorder=12)

    # Decoration
    ax.set_xlabel('\nTime (s)', fontsize=12, linespacing=3.0)
    ax.set_ylabel('\nTemp (°C)', fontsize=12, linespacing=3.0)
    ax.set_zlabel('\nEquiv. Ratio', fontsize=12, linespacing=3.0)
    ax.set_title(f'Standard BO Trajectory (Fixed Kernel)\n(Max Background Yield: {max_y:.2f} M)', fontsize=14, pad=20)

    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.view_init(elev=20, azim=-50)

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label('Background Yield (M)', fontsize=10)

    ax.legend(loc='upper left')
    plt.tight_layout()
    try:
        plt.savefig(filename, dpi=300, transparent=True)
    except:
        plt.savefig(filename, dpi=300)
    plt.show()
    print(f"Done! Image saved as '{filename}'")


if __name__ == "__main__":
    run_standard_bo()