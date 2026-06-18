# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import GPy
from scipy.stats import qmc

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path: sys.path.insert(0, current_dir)

import config_robertson as cfg
from robertson_generalized import RobertsonGeneralized
from cigp_model_normalized import CIGP_Normalized


def generate_mixed_noise_data(true_model):
    print("[Data] Generating LHS training data (3600s, Product C)...")
    t_start, t_end = cfg.DATA_GEN_CONFIG['TIME_SPAN']
    n_total = cfg.DATA_GEN_CONFIG['N_TOTAL_POINTS']

    # LHS 采样时间
    sampler = qmc.LatinHypercube(d=1)
    sample = sampler.random(n=n_total)
    X_train = t_start + sample * (t_end - t_start)
    X_train = np.sort(X_train, axis=0)

    # 计算真值
    phys_truth = true_model.truth_params_phys
    norm_truth = (phys_truth - true_model.lb) / true_model.diff
    Y_true = true_model.compute_mean(X_train, norm_truth)

    # 混合噪声
    n_high = cfg.DATA_GEN_CONFIG['N_HIGH_NOISE']
    indices = np.arange(n_total)
    np.random.shuffle(indices)
    high_noise_idx = indices[:n_high]

    noise = np.zeros_like(Y_true)
    peak = np.max(Y_true) + 1e-9

    noise += np.random.normal(0, cfg.DATA_GEN_CONFIG['NOISE_LOW'] * peak, size=Y_true.shape)
    noise[high_noise_idx] += np.random.normal(0, cfg.DATA_GEN_CONFIG['NOISE_HIGH'] * peak, size=(n_high, 1))

    Y_obs = Y_true + noise
    Y_obs = np.maximum(Y_obs, 0.0)

    return X_train, Y_obs, Y_true


def get_normalized_lhs_samples(n_samples):
    n_dim = len(cfg.CURRENT_PARAM_NAMES)
    sampler = qmc.LatinHypercube(d=n_dim)
    return sampler.random(n=n_samples)


def main():
    print(">>> Benchmark: Parameter Inversion (3600s)")
    SCALE = cfg.BENCHMARK_SETTINGS['SCALE_FACTOR']

    # 结构误差（Kernel）方差限制
    VAR_LIMIT = cfg.BENCHMARK_SETTINGS['GP_VAR_LIMIT']
    # 噪声（Likelihood）方差限制 - 必须极低才能强迫经过零点
    # 如果 config 里没写，默认用这个极低范围
    NOISE_LIMIT = cfg.BENCHMARK_SETTINGS.get('GP_NOISE_VAR_LIMIT', [1e-7, 1e-6])

    # 1. 准备数据
    true_model = RobertsonGeneralized(mode='true', scale_factor=SCALE)
    X_train, Y_train, Y_true_train = generate_mixed_noise_data(true_model)

    # ==========================================
    # [FIX 1] 注入零点约束
    # ==========================================
    print(f"[Data] Injecting Zero Constraints (Noise Limit: {NOISE_LIMIT})...")
    n_anchors = 5
    X_zero = np.linspace(0, 1e-6, n_anchors).reshape(-1, 1)
    Y_zero = np.zeros((n_anchors, 1))

    # 保存原始真实数据用于计算 scaler
    Y_train_real = Y_train.copy()

    # 拼接到训练数据
    X_train = np.vstack((X_zero, X_train))
    Y_train = np.vstack((Y_zero, Y_train))

    # ==========================================
    # [FIX - Qwen Suggestion] 鲁棒归一化
    # ==========================================
    # 不要用包含 0 的 Y_train 来计算均值方差，只用真实数据计算
    # 这样能保证物理模型在主要工作区间（700-800）的归一化是准确的
    y_mean = np.mean(Y_train_real)
    y_std = np.std(Y_train_real)

    if y_std < 1e-9: y_std = 1.0

    # 用真实的统计量去归一化整个数据集（包括锚点）
    Y_norm = (Y_train - y_mean) / y_std
    Y_scaler = {'mean': y_mean, 'std': y_std}
    # ==========================================

    y_mean, y_std = np.mean(Y_train), np.std(Y_train)
    if y_std < 1e-9: y_std = 1.0
    Y_norm = (Y_train - y_mean) / y_std

    # 2. Standard GP
    print("\n[1/2] Training Standard GP...")
    # 使用 Matern32 以适应起点的剧烈变化
    k_std = GPy.kern.Matern32(1, variance=1.0, lengthscale=cfg.BENCHMARK_SETTINGS['GP_LEN_INIT'])
    m_gp = GPy.models.GPRegression(X_train, Y_norm, k_std)
    m_gp.optimize_restarts(cfg.BENCHMARK_SETTINGS['GP_RESTARTS'], verbose=False)

    # 3. CIGP (LHS Optimization)
    print("\n[2/2] Training CIGP (LHS Search)...")
    lhs_n = cfg.BENCHMARK_SETTINGS['LHS_OPT_SAMPLES']
    guesses = get_normalized_lhs_samples(lhs_n)
    Y_scaler = {'mean': y_mean, 'std': y_std}

    best_cigp = None
    best_ll = -np.inf

    for i, w0_norm in enumerate(guesses):
        try:
            phys = RobertsonGeneralized(mode='variable', scale_factor=SCALE, w_init=w0_norm)

            # [FIX 2] 更换核函数: Matern32 比 RBF 更适合拟合 "0 -> 0.03" 的突变
            k_cigp = GPy.kern.Matern32(1, variance=0.01, lengthscale=cfg.BENCHMARK_SETTINGS['GP_LEN_INIT'])

            # 限制结构误差幅度 (Kernel Variance)
            k_cigp.variance.constrain_bounded(*VAR_LIMIT)

            m = CIGP_Normalized(X_train, Y_norm, phys, Y_scaler, kernel=k_cigp)
            m.W.constrain_bounded(0.0, 1.0)

            # ======================================================
            # [FIX 3] 强制生效的噪声约束 (CRITICAL FIX)
            # ======================================================
            if hasattr(m, 'likelihood') and hasattr(m.likelihood, 'variance'):
                # 1. 先手动赋值！把 1.0 变成 5e-7 (假如 limit 是 1e-7~1e-6)
                # 这一步是为了让参数“瞬移”到合法区间，否则 constrain_bounded 会失败或被忽略
                mid_val = (NOISE_LIMIT[0] + NOISE_LIMIT[1]) / 2.0
                m.likelihood.variance[:] = mid_val

                # 2. 再加约束，此时当前值已经合法，约束就会稳固生效
                m.likelihood.variance.constrain_bounded(NOISE_LIMIT[0], NOISE_LIMIT[1], warning=False)

            # 优化 (robust=True 增加数值稳定性)
            m.optimize(messages=False, max_iters=cfg.BENCHMARK_SETTINGS['OPT_MAX_ITER'])

            # 记录最佳结果
            if m.log_likelihood() > best_ll:
                best_ll, best_cigp = m.log_likelihood(), m
                # 可选：打印 debug 信息确认噪声值
                # curr_noise = m.likelihood.variance.values[0]
                # print(f"  > Iter {i+1}: LL={best_ll:.2f}, Noise={curr_noise:.2e}")
                print(f"  > Iter {i + 1}: New best LL = {best_ll:.4f}")

        except Exception as e:
            # 忽略个别优化失败的情况
            pass

    if best_cigp is None:
        print("[Error] Optimization Failed. All initial guesses resulted in errors.")
        return

    # 4. 绘图
    t_start, t_end = cfg.DATA_GEN_CONFIG['TIME_SPAN']
    X_test = np.linspace(0, t_end, 200).reshape(-1, 1)

    mu_gp = (m_gp.predict(X_test)[0] * y_std + y_mean)
    mu_cigp = (best_cigp.predict(X_test)[0] * y_std + y_mean)

    phys_truth = true_model.truth_params_phys
    norm_truth = (phys_truth - true_model.lb) / true_model.diff
    Y_true_curve = true_model.compute_mean(X_test, norm_truth)

    raw_phys = best_cigp.physics_model.compute_mean(X_test, best_cigp.W.values)
    g_err_real = mu_cigp - raw_phys

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    ax1.plot(X_test, Y_true_curve, 'k-', label='Ground Truth', linewidth=1.5, alpha=0.4)

    # 绘图：去掉前 5 个锚点避免重叠，单独画一个红叉
    ax1.plot(X_train[n_anchors:], Y_train[n_anchors:], 'ko', label='Training Data', markersize=5)
    ax1.plot(0, 0, 'rx', label='Zero Constraint', markersize=8, markeredgewidth=2)

    ax1.plot(X_test, mu_gp, 'b--', label='Standard GP', linewidth=1.5)
    ax1.plot(X_test, mu_cigp, 'r-', label='CIGP (Hybrid)', linewidth=2.5)
    ax1.plot(X_test, raw_phys, 'r:', label='CIGP (Phys Model)', linewidth=1.5)

    ax1.set_ylabel('Concentration [C]')
    ax1.set_title(f'Benchmark: Parameter Inversion\nForce Zero Constraint: Noise Limit {NOISE_LIMIT}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(X_test, g_err_real, 'purple', label='Structural Error ($g_{err}$)', linewidth=1.5)
    ax2.fill_between(X_test.flatten(), 0, g_err_real.flatten(), color='purple', alpha=0.1)
    ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax2.set_ylabel('Correction')
    ax2.set_xlabel('Time (seconds)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("benchmark_optimization_result.png", dpi=300)
    plt.show()

    # 5. 打印报告
    print("\n" + "=" * 60)
    print("      CIGP 反演结果 (Parameter Inversion)")
    print("=" * 60)

    # 验证最终噪声值
    final_noise = best_cigp.likelihood.variance.values[0]
    print(f"[Debug] Final Likelihood Noise: {final_noise:.2e} (Constraint: {NOISE_LIMIT})")

    w_norm_fit = best_cigp.W.values
    w_phys_fit = best_cigp.physics_model._unscale_params(w_norm_fit)

    k_fit = 10 ** w_phys_fit[:4]
    n_fit = w_phys_fit[4:]
    k_true = 10 ** true_model.truth_params_phys[:4]
    n_true = true_model.truth_params_phys[4:]

    print(f"{'Param':<10} | {'True':<10} | {'Fitted':<10} | {'Error'}")
    print("-" * 60)
    for i in range(4):
        err = abs(k_fit[i] - k_true[i]) / k_true[i]
        print(f"k_{i + 1:<8} | {k_true[i]:<10.2e} | {k_fit[i]:<10.2e} | {err:.1%}")
    print("-" * 60)
    for i in range(4):
        err = abs(n_fit[i] - n_true[i]) / n_true[i]
        print(f"n_{i + 1:<8} | {n_true[i]:<10.2f} | {n_fit[i]:<10.2f} | {err:.1%}")
    print("=" * 60)
    print(f"\n[拟合方程]: Rate2 = {k_fit[1]:.2e} * [B]^{n_fit[1]:.2f}")


if __name__ == "__main__":
    main()