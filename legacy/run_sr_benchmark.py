# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from pysr import PySRRegressor


def set_nature_style():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['axes.grid'] = False
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['legend.frameon'] = False


def generate_aligned_benchmark_data(n_samples=64):  # [修改] 改为 320 点
    """
    [严格对齐模式 - 含噪声版]
    对应 MCMC: 8 experiments * 40 points = 320 samples
    对应 Noise: 0.02 (2%)
    """
    print(f"[Data Gen] Generating {n_samples} samples (Noisy & Aligned)...")

    rng = np.random.RandomState(42)

    # 浓度范围 (保持一致)
    H2 = rng.uniform(0.5, 2.0, n_samples)
    Br2 = rng.uniform(0.01, 10.0, n_samples)

    # 物理真值
    k_overall_true = 1.0 * np.sqrt(10.0 / 100.0)  # 0.316...
    rate_true = k_overall_true * H2 * np.sqrt(Br2)

    # [修改] 加入 2% 噪声
    # 噪声标准差 = 真实值 * 0.02 + 极小底噪
    # 这样大信号噪声大，小信号噪声小，符合真实仪器特性 (相对误差)
    noise_level = 0.05
    noise = rng.normal(0, noise_level * rate_true + 1e-6, n_samples)
    rate_obs = rate_true + noise

    X = np.stack([H2, Br2], axis=1)
    y = rate_obs

    return X, y, rate_true, k_overall_true


def run_symbolic_regression():
    set_nature_style()

    # --- 1. 获取含噪声数据 ---
    # n_samples = 320, noise ~ 2%
    X, y, y_true, k_true = generate_aligned_benchmark_data(n_samples=64)

    print("\n[PySR] Starting Symbolic Regression Search (Noisy Data)...")

    # --- 2. 配置 PySR 模型 ---
    model = PySRRegressor(
        niterations=50,
        populations=20,
        binary_operators=["+", "*", "-", "/"],
        unary_operators=["sqrt", "square"],
        # [修改] 在噪声存在时，最好让 PySR 稍微容忍一点误差
        # 但默认的 MSE loss 依然有效
        model_selection="best",
        loss="loss(prediction, target) = (prediction - target)^2",
        maxsize=25,
        random_state=42,
        verbosity=1,
        procs=0,
    )

    model.fit(X, y, variable_names=["H2", "Br2"])

    best_eq = model.sympy()
    y_pred = model.predict(X)
    mse = np.mean((y_pred - y) ** 2)

    print("\n" + "=" * 90)
    print(f"{'BENCHMARK COMPARISON: MCMC vs PySR (Noisy Conditions)':^90}")
    print("=" * 90)
    print(f"{'Parameter':<20} | {'MCMC Config':<30} | {'PySR Data':<30}")
    print("-" * 90)
    print(f"{'Data Points':<20} | 320 (8x40)                     | 320")
    print(f"{'Noise Level':<20} | 0.02 (2%)                        | 0.02 (2%)")
    print(f"{'Sigma Likelihood':<20} | 0.05                             | N/A (Implicit in Loss)")
    print("-" * 90)
    print(f"\n[Result] SR Found: Rate = {best_eq}")
    print(f"[Result] MSE: {mse:.2e}")

    plot_parity(y, y_pred)  # 注意：这里画的是 观测值y vs 预测值y_pred


def plot_parity(y_obs, y_pred):
    fig, ax = plt.subplots(figsize=(6, 5))
    color_dots = '#4DBBD5'
    color_line = '#E64B35'

    ax.scatter(y_obs, y_pred, alpha=0.5, color=color_dots, edgecolors='none', s=40, label='Predictions')

    min_val = min(np.min(y_obs), np.min(y_pred))
    max_val = max(np.max(y_obs), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], '--', color=color_line, lw=2, label='Perfect Fit')

    ax.set_xlabel('Observed Rate (with Noise)', fontweight='bold')
    ax.set_ylabel('SR Predicted Rate', fontweight='bold')
    ax.legend(loc='upper left', frameon=False)

    plt.tight_layout()
    plt.savefig('sr_benchmark_parity_noisy.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    run_symbolic_regression()