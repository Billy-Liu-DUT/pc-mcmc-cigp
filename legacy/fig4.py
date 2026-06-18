import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

# ==========================================
# 1. 数据准备 (Data Preparation)
# ==========================================
# 迭代轮数 (LHS=10, BO=15)
iter_nums = np.arange(0, 16)  # 0代表LHS结束时的状态
iters_1_to_15 = np.arange(1, 16)

# --- Standard BO Data ---
# LHS最佳: 0.6018
std_bo_obs = [0.0382, 0.1516, 0.5248, 0.1420, 0.6103, 0.4587, 0.5572, 0.5744, 0.4820, 0.5530, 0.6188, 0.3558, 0.5733, 0.1239, 0.6925]
std_bo_best_history = [0.6018]
current_best = 0.6018
for y in std_bo_obs:
    current_best = max(current_best, y)
    std_bo_best_history.append(current_best)

# --- CIGP Data ---
# LHS最佳: 0.5691
cigp_obs = [0.6409, 0.4961, 0.3866, 0.7278, 0.7788, 0.7658, 0.7586, 0.7404, 0.7280, 0.7362, 0.7302, 0.7475, 0.7495, 0.7360, 0.7615]
cigp_best_history = [0.5691]
current_best = 0.5691
for y in cigp_obs:
    current_best = max(current_best, y)
    cigp_best_history.append(current_best)

# ==========================================
# 2. 绘图设置 (Plotting Style)
# ==========================================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5

# --- 配色方案 (Color Scheme) ---
# Standard BO: 红色 (保持不变)
color_std = '#D62728'
# CIGP: 浅蓝色 (Lighter Blue, e.g., DodgerBlue or similar)
color_cigp = '#42A5F5'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

# -------------------------------------------------------
# 子图 1: 最佳产率追踪 (Best-so-far Curve)
# -------------------------------------------------------
# 绘制 Standard BO
ax1.plot(iter_nums, std_bo_best_history, marker='^', markersize=8, linestyle='-', linewidth=2.5,
         color=color_std, label='Standard BO (Benchmark)', alpha=0.8)

# 绘制 CIGP
ax1.plot(iter_nums, cigp_best_history, marker='o', markersize=8, linestyle='-', linewidth=2.5,
         color=color_cigp, label='CIGP (Proposed)', zorder=10)

# 辅助线
ax1.axhline(y=0.79, color='green', linestyle='--', linewidth=1.5, alpha=0.6, label='Theoretical Max (~0.79 M)')

# 装饰
ax1.set_xlabel('BO Iterations', fontweight='bold')
ax1.set_ylabel('Best Yield Found (M)', fontweight='bold')
ax1.set_title('(a) Optimization Efficiency', fontweight='bold', pad=15)
ax1.legend(frameon=True, loc='lower right', fontsize=10)
ax1.grid(True, which='major', linestyle='--', alpha=0.4)
ax1.set_xlim(0, 15)
ax1.set_ylim(0.55, 0.82)

# -------------------------------------------------------
# 子图 2: 实验点分布 (Observed Yield per Batch)
# -------------------------------------------------------
# 【修改点】：使用 plot 加上 linestyle='-' 来连接点
# Standard BO (实线连接，但用空心点或不同透明度表示震荡)
ax2.plot(iters_1_to_15, std_bo_obs, color=color_std, marker='^', markersize=8, linestyle='-', linewidth=1.5,
         label='Standard BO Obs.', alpha=0.7, markerfacecolor='white', markeredgewidth=2)

# CIGP (实线连接，实心点)
ax2.plot(iters_1_to_15, cigp_obs, color=color_cigp, marker='o', markersize=8, linestyle='-', linewidth=2.5,
         label='CIGP Obs.', alpha=0.9, zorder=10)

# 危险/低效区域背景
ax2.fill_between([0, 16], 0, 0.2, color='gray', alpha=0.1, label='Inefficient Zone (<0.2 M)')

# 装饰
ax2.set_xlabel('BO Iterations', fontweight='bold')
ax2.set_ylabel('Observed Yield per Batch (M)', fontweight='bold')
ax2.set_title('(b) Process Stability & Safety', fontweight='bold', pad=15)
ax2.legend(loc='lower right', fontsize=10)
ax2.grid(True, which='major', linestyle='--', alpha=0.4)
ax2.set_xlim(0.5, 15.5) # 稍微调整范围让点不贴边
ax2.set_ylim(0, 0.85)

# ==========================================
# 3. 保存与显示
# ==========================================
plt.tight_layout()
plt.savefig('comparison_plot_v2.png', dpi=300, bbox_inches='tight')
plt.show()