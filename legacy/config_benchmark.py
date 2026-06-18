# -*- coding: utf-8 -*-
import numpy as np

"""
==================================================================================
CIGP Benchmark 配置文件 (Final Version)
==================================================================================
"""

# --- 1. 贝叶斯优化流程控制 ---
N_LHS_SAMPLES = 10        # 外层：初始实验点数
N_BO_ITERS    = 15       # 外层：贝叶斯优化轮数

# 【新增】内层 ODE 拟合的重启次数
# 设为 1: 极速模式 (仅使用上一轮结果热启动)
# 设为 5-10: 稳健模式 (1次热启动 + 4-9次LHS全局搜索)
N_INNER_RESTARTS = 1

# --- 【新增】全域测试集设置 (用于验证外推能力) ---
N_TEST_SAMPLES = 200     # 测试点数量
TEST_SEED      = 42     # 固定种子，保证 CIGP 和 Standard BO 考的是同一套题

# 采集函数
ACQ_FUNC_TYPE = 'PC_EI'  # 对应 acquisition_functions.py 中的 PhysConstrainedEI
ACQ_PARAMS    = {
    'xi': 0.6,          # 基础 EI 的探索参数
   'threshold': 0.1,    # 【门槛】物理预测产率低于 0.1M (10%) 的地方，权重降低
   'sharpness': 5.0     # 【陡峭度】Sigmoid 函数的斜率，控制“封杀”的果断程度
}
# --- 2. 物理/化学空间定义 (0.8 - 1.2 M) ---
DESIGN_SPACE = {
    'names': ['C_Styrene', 'C_PAA', 'Temperature', 'Time'],
    'bounds': np.array([
        [0.8, 1.2],    # C_Styrene
        [0.8, 1.2],    # C_PAA
        [303.0, 413.0],# T (30-140C)
        [60.0, 3600.0] # t
    ])
}

# --- 3. CIGP 模型配置 ---
PHYSICS_OPERATOR = 'EPOXY_VAR'
OPTIMIZER = 'scg'
MAX_ITERS = 500

# --- 4. 物理参数边界 ---
PHYSICS_BOUNDS = {
    'alpha_1': [0.95, 1.05], 'beta_1': [0.95, 1.05],
    'log10_A_1': [5.0, 7.0], 'log10_Ea_1': [4.0, 5.0],
    'alpha_2': [0.95, 1.05], 'beta_2': [0.95, 1.05],
    'log10_A_2': [9.0, 11.0], 'log10_Ea_2': [4.5, 5.0]
}

# GP 核函数与噪声约束
GP_CONST = {
    'noise_var': [1e-6, 0.001],
    'kernel_var': [1e-6, 0.01],
    'kernel_len': 0.5
}

# --- 5. 可视化保存路径 ---
VIS_SAVE_PATH = 'benchmark_final_vis.png'