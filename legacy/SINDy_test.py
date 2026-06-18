# -*- coding: utf-8 -*-
import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# ==========================================
# 1. 真实物理系统 (Ground Truth)
# ==========================================
def reaction_rate(C, t, k_true):
    H2, Br2, HBr, H, Br = C
    k1, k2, k3, k4, k5 = k_true
    r1 = k1 * Br2
    r2 = k2 * Br ** 2
    r3 = k3 * Br * H2
    r4 = k4 * H * Br2
    r5 = k5 * H * HBr

    dH2 = -r3 + r5
    dBr2 = -r1 + r2 - r4 + r5
    dHBr = r3 + r4 - r5
    dH = r3 - r4 - r5
    dBr = 2 * r1 - 2 * r2 - r3 + r4 + r5
    return [dH2, dBr2, dHBr, dH, dBr]


k_ground_truth = [50.0, 100.0, 1.0, 50.0, 10.0]

# ==========================================
# 2. 生成稀疏含噪数据
# ==========================================
np.random.seed(42)
t_span = np.linspace(0, 5, 20)
dt = t_span[1] - t_span[0]
C0 = [1.0, 1.0, 0.0, 0.0, 0.0]

C_true = odeint(reaction_rate, C0, t_span, args=(k_ground_truth,))
noise_level = 0.05
C_noisy = C_true * (1 + noise_level * np.random.randn(*C_true.shape))

# ==========================================
# 3. [核心修复] 手动计算导数 (Bypass Numpy Bug)
# ==========================================
# 自己用 numpy 算导数，不让 PySINDy 算，这样就不会报错
print("Computing derivatives manually...")
x_dot_precomputed = np.gradient(C_noisy, dt, axis=0)

# ==========================================
# 4. 配置 SINDy
# ==========================================
feature_library = ps.CustomLibrary(
    library_functions=[
        lambda x: x,
        lambda x, y: x * y,
        lambda x, y: x * (np.maximum(y, 0) ** 0.5)
    ],
    function_names=[
        lambda x: x,
        lambda x, y: f"{x}{y}",
        lambda x, y: f"{x}{y}^0.5"
    ]
)

optimizer = ps.STLSQ(threshold=0.1, alpha=0.05)
model = ps.SINDy(feature_library=feature_library, optimizer=optimizer)

# ==========================================
# 5. 训练模型
# ==========================================
print("-" * 60)
print("SINDy Identification Result (Baseline)")
print("-" * 60)

# [关键改动] 传入 x_dot，PySINDy 就会跳过内部微分步骤，直接开始回归
model.fit(C_noisy, x_dot=x_dot_precomputed, t=dt)

try:
    model.print()
except:
    pass

# ==========================================
# 6. 可视化 (复刻 Reference 矩形风格)
# ==========================================
print("Running simulation...")

try:
    # 尝试积分，如果方程离谱可能会报错，那是正常的（我们要展示的就是它不行）
    C_sindy_pred = model.simulate(C0, t_span, integrator_kws={'method': 'Radau'})
except Exception as e:
    print(f"Simulation crashed (Expected behavior): {e}")
    C_sindy_pred = np.zeros((1, 5))

# 如果发散（点数不够），补 NaN
if C_sindy_pred.shape[0] < len(t_span):
    print("Model diverged! Padding with NaNs.")
    C_full = np.full((len(t_span), 5), np.nan)
    C_full[:C_sindy_pred.shape[0], :] = C_sindy_pred
    C_sindy_pred = C_full

# --- 绘图配置 ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

COLOR_TRUE = '#1B9E77'
COLOR_DATA = '#66C2A5'
COLOR_SINDY = '#D55E00'

# 设置为矩形比例 (6, 5) 匹配参考图
plt.figure(figsize=(6, 5))

# Ground Truth
plt.plot(t_span, C_true[:, 2], color=COLOR_TRUE, linewidth=2.5,
         label='Ground Truth', zorder=2)

# SINDy (发散/错误)
plt.plot(t_span, C_sindy_pred[:, 2], color=COLOR_SINDY, linestyle='--', linewidth=2.5,
         label='SINDy (Unstable)', zorder=3)

# Noisy Data
plt.scatter(t_span, C_noisy[:, 2], color=COLOR_DATA, s=50, alpha=0.8,
            label='Noisy Data', zorder=4, edgecolors='none')

plt.xlabel('Time (s)', fontweight='bold', fontsize=14)
plt.ylabel('Concentration (M)', fontweight='bold', fontsize=14)

ax = plt.gca()
ax.tick_params(direction='in', length=5, width=1.2, top=True, right=True)

plt.ylim(bottom=-0.5, top=np.max(C_noisy[:, 2]) * 1.15)
plt.xlim(left=0, right=t_span[-1] * 1.05)
plt.legend(frameon=False, fontsize=11, loc='best')
plt.tight_layout()

plt.savefig('sindy_benchmark_fixed_final.png', dpi=300, bbox_inches='tight')
plt.show()