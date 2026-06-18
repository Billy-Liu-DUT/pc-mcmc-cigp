# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
import matplotlib.colors as colors
import matplotlib.cm as cm

# ==========================================
# 1. 物理模型与参数 (Ground Truth)
# ==========================================
R_GAS = 8.314
GT_PARAMS = {
    'Ea1': 55000.0, 'logA1': 6.0,  # 主反应
    'Ea2': 85000.0, 'logA2': 10.0  # 副反应 (自毒化)
}


def _ode_kinetics(t, y, k1, k2):
    S, PAA, E, Acid = y
    S, PAA, E, Acid = [max(0, x) for x in [S, PAA, E, Acid]]
    r1 = k1 * S * PAA
    r2 = k2 * E * Acid
    return [-r1, -r1, r1 - r2, r1 - r2]


def get_yield_constrained(time, temp, ratio):
    """
    根据当量比计算产率，同时严格遵守 0.8 - 1.2 M 的浓度限制。
    策略：为了模拟最佳工况，我们在满足比例的前提下，
    让限制性底物的浓度尽可能大（顶格到 1.2 M）。
    """
    # 浓度边界
    C_MIN, C_MAX = 0.8, 1.2

    if ratio >= 1.0:
        # PAA 过量或等量。让 PAA 顶格到 1.2
        c_paa = C_MAX
        c_sty = c_paa / ratio
        # 检查 Styrene 是否低于下限 0.8
        if c_sty < C_MIN:
            # 如果低于下限，则被迫提高 Styrene 到 0.8，PAA 相应增加（但这会由 Ratio 决定，产生矛盾）
            # 正确逻辑：Ratio = PAA/Sty。
            # Max Ratio = 1.2/0.8 = 1.5. 所以只要 Ratio <= 1.5，上述逻辑 c_sty 都在 [0.8, 1.2] 内。
            pass
    else:
        # Styrene 过量。让 Styrene 顶格到 1.2
        c_sty = C_MAX
        c_paa = c_sty * ratio
        # 同理，Min Ratio = 0.8/1.2 = 0.667。只要 Ratio >= 0.67，c_paa 就在 [0.8, 1.2] 内。

    # 计算速率常数
    k1 = (10 ** GT_PARAMS['logA1']) * np.exp(-GT_PARAMS['Ea1'] / (R_GAS * temp))
    k2 = (10 ** GT_PARAMS['logA2']) * np.exp(-GT_PARAMS['Ea2'] / (R_GAS * temp))

    y0 = [c_sty, c_paa, 0.0, 0.0]

    try:
        sol = solve_ivp(_ode_kinetics, [0, time], y0, args=(k1, k2),
                        method='LSODA', rtol=1e-3, atol=1e-3)
        return sol.y[2, -1]
    except:
        return 0.0


# ==========================================
# 2. 生成 3D 网格数据 (修正范围)
# ==========================================
print("正在计算反应空间数据 (浓度范围 0.8-1.2 M)...")

n_t = 30  # 时间轴采样
n_T = 30  # 温度轴采样
n_r = 20  # 当量比轴采样

# 轴范围 (根据 0.8-1.2 M 修正)
t_range = np.linspace(60, 3600, n_t)  # Time: 1min - 60min
T_range = np.linspace(303, 413, n_T)  # Temp: 30 - 140 C
r_range = np.linspace(0.67, 1.5, n_r)  # Ratio: 0.67 - 1.5 (严格卡死边界)

grid_t, grid_T, grid_r = np.meshgrid(t_range, T_range, r_range, indexing='ij')
grid_yield = np.zeros_like(grid_t)

total_points = grid_t.size
count = 0
for i in range(n_t):
    for j in range(n_T):
        for k in range(n_r):
            grid_yield[i, j, k] = get_yield_constrained(grid_t[i, j, k], grid_T[i, j, k], grid_r[i, j, k])
            count += 1
            if count % 2000 == 0:
                print(f"进度: {count}/{total_points}")

max_y = np.max(grid_yield)
print(f"最大产率: {max_y:.4f} M")

# ==========================================
# 3. 顶刊风格 3D 绘图 (单图)
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei', 'DejaVu Sans']

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 展平数据
xs = grid_t.flatten()
ys = grid_T.flatten() - 273.15  # 转摄氏度
zs = grid_r.flatten()
vals = grid_yield.flatten()

# 1. 过滤：只显示有意义的产率区域 (例如 > 最大值的 10%)
# 由于浓度很高，反应很快，为了看清核心，我们过滤掉非常低产率的点
mask = vals > (max_y * 0.1)
xs, ys, zs, vals = xs[mask], ys[mask], zs[mask], vals[mask]

# 2. 归一化 & 配色
norm = colors.Normalize(vmin=0, vmax=max_y)
# 使用 'Spectral_r' (红=高, 蓝=低) 或 'plasma' (亮黄=高, 紫=低)
# Spectral_r 在白色背景下对比度很好
cmap = plt.get_cmap('Spectral_r')
rgba_colors = cmap(norm(vals))

# 3. 透明度映射 (Alpha Mapping)
# 核心逻辑：产率越高越不透明。
# 使用 Power Law (幂次) 让低值区更透明，突出高值核心
val_normalized = vals / max_y
alphas = val_normalized ** 3  # 3次方会让 0.5 变成 0.125(透明), 0.9 变成 0.72(可见)
alphas = np.clip(alphas, 0.05, 0.95)  # 限制范围
rgba_colors[:, 3] = alphas

# 4. 绘制散点
sc = ax.scatter(xs, ys, zs,
                c=rgba_colors,
                s=80,  # 点大一点，更有体积感
                marker='o',
                edgecolor='none',
                depthshade=True)  # 开启深度阴影

# --- 坐标轴美化 ---
# 加大字号
label_fs = 14
tick_fs = 11

ax.set_xlabel('\nTime (s)', fontsize=label_fs, linespacing=3.0)
ax.set_ylabel('\nTemperature (°C)', fontsize=label_fs, linespacing=3.0)
ax.set_zlabel('\nEquiv. Ratio (PAA/Sty)', fontsize=label_fs, linespacing=3.0)
ax.set_title('Reaction Yield Landscape (Concentration 0.8-1.2 M)', fontsize=16, pad=20)

ax.tick_params(axis='both', which='major', labelsize=tick_fs)

# 去除背景灰色
ax.grid(False)
ax.xaxis.pane.fill = False;
ax.yaxis.pane.fill = False;
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('w');
ax.yaxis.pane.set_edgecolor('w');
ax.zaxis.pane.set_edgecolor('w')

# 调整视角 (Elev=俯仰角, Azim=方位角)
# 这个角度能看清 Ratio 和 Temp 的关系
ax.view_init(elev=20, azim=-50)

# --- Colorbar ---
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
cbar.set_label('Yield (M)', fontsize=14)
cbar.ax.tick_params(labelsize=tick_fs)

plt.tight_layout()
plt.savefig('reaction_cloud_corrected.png', dpi=300, transparent=True)
plt.show()

print("绘图完成！已保存为 reaction_cloud_corrected.png")