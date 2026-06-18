# -*- coding: utf-8 -*-
import numpy as np

"""
================================================================
Robertson Benchmark Config (Optimized for mmol/L units)
================================================================
"""

# 1. 物理定义 (保持不变)
SPECIES_DEFINITIONS = {
    'A': {'Z': 1, 'Charge': 0},
    'B': {'Z': 1, 'Charge': 0},
    'C': {'Z': 1, 'Charge': 0},
    'D': {'Z': 1, 'Charge': 0},
}

# 2. 核心控制参数 (优化时间尺度和噪声限制)
BENCHMARK_SETTINGS = {
    'SCALE_FACTOR': 1.0,           # 1:1时间比例 (秒)
    'GP_VAR_LIMIT': [1e-6, 1.0],   # 结构误差方差限制
    'GP_LEN_INIT': 500.0,          # GP长度尺度 (适应3600秒范围)
    'GP_NOISE_VAR_LIMIT': [1e-8, 1e-6],  # 关键：极低噪声限制
    'LHS_OPT_SAMPLES': 30,         # 增加LHS采样数
    'OPT_MAX_ITER': 3000,          # 增加优化迭代次数
    'GP_RESTARTS': 10,             # 增加GP重启次数
}

# 3. 数据生成设置 (使用mmol/L单位)
DATA_GEN_CONFIG = {
    'N_TOTAL_POINTS': 15,         # 总训练点数
    'N_HIGH_NOISE': 5,            # 高噪声点数
    'NOISE_HIGH': 0.10,            # 10% 高噪声
    'NOISE_LOW': 0.02,             # 2% 低噪声
    'TIME_SPAN': [0, 3600],        # 0到3600秒
    'INITIAL_CONC': [1000.0, 0.0, 0.0]  # 初始浓度 [A, B, C] in mmol/L
}

# 4. ODE 求解器设置 (保持刚性求解)
ODE_METHOD = 'LSODA'   # 适合刚性问题的求解器
ODE_RTOL = 1e-8        # 相对误差
ODE_ATOL = 1e-10       # 绝对误差

# 5. 参数边界库 (关键修改：优化边界以适应mmol/L单位)
_BOUNDS_LIB = {
    'ROBERTSON_GEN': {
        'names': ['log_k1', 'log_k2', 'log_k3', 'log_k4', 'n1', 'n2', 'n3', 'n4'],
        'bounds': {
            # k1 (一级): 不变 (真值 -0.4)
            'log_k1': [-2.0, 1.0],

            # k2 (二级): 减去 3.0 (真值 8.48 -> 5.48)
            # 原始 [7.0, 10.0] -> 新范围 [4.0, 7.0]
            'log_k2': [4.0, 7.0],

            # k3 (二级): 减去 3.0 (真值 5.0 -> 2.0)
            # 原始 [4.0, 6.0] -> 新范围 [1.0, 3.0]
            'log_k3': [1.0, 3.0],

            # k4 (一级): 不变 (真值 -8.0)
            'log_k4': [-10.0, -6.0],

            # 反应级数不变
            'n1': [0.9, 1.1],
            'n2': [1.9, 2.1],
            'n3': [0.9, 1.1],
            'n4': [0.9, 1.1]
        }
    }
}

# 6. 模型配置
PHYSICS_MODEL_TYPE = 'ROBERTSON_GEN'
W_BOUNDS = _BOUNDS_LIB['ROBERTSON_GEN']['bounds']
CURRENT_PARAM_NAMES = _BOUNDS_LIB['ROBERTSON_GEN']['names']