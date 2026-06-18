# -*- coding: utf-8 -*-
import numpy as np

"""
==================================================================================
CIGP 配置文件 (v28.0 5-Operator Zoo)
==================================================================================
"""

# 1. 数据集
N_DATA = 25

# 2. 优化设置
N_RESTARTS = 50
NUM_CORES = 'auto'
OPTIMIZER = 'scg'
MAX_ITERS = 1000

# 3. ODE 求解器
# 强烈推荐 LSODA，特别是对于自催化和连串反应
ODE_METHOD = 'LSODA'
ODE_RTOL = 1e-2
ODE_ATOL = 1e-3

# ==========================================
# 4. 算子选择开关 (Model Selector)
# ==========================================
# 可选: 'SIMPLE', 'REVERSIBLE_VAR', 'PARALLEL_VAR', 'SERIES_VAR', 'AUTOCAT_VAR'
# ==========================================
PHYSICS_MODEL_TYPE = 'SERIES_VAR'  # <--- 在这里修改！

# ==========================================
# 5. 参数边界库
# ==========================================
_BOUNDS_LIB = {
    'SIMPLE': {
        'names': ['alpha', 'beta', 'log10_A', 'log10_Ea'],
        'bounds': {
            'alpha': [0., 2.], 'beta': [0., 2.],
            'log10_A': [3., 8.], 'log10_Ea': [4., 5.]
        }
    },
    'REVERSIBLE_VAR': {
        'names': ['alpha', 'beta', 'gamma', 'log10_A_fwd', 'log10_Ea_fwd', 'log10_A_rev', 'log10_Ea_rev'],
        'bounds': {
            'alpha': [0.5, 3.], 'beta': [0.5, 3.], 'gamma': [0.5, 3.],
            'log10_A_fwd': [3., 8.], 'log10_Ea_fwd': [4., 6.],
            'log10_A_rev': [2., 7.], 'log10_Ea_rev': [4., 6.]
        }
    },
    'PARALLEL_VAR': {
        'names': ['alpha_m', 'beta_m', 'log10_A_m', 'log10_Ea_m',
                  'alpha_s', 'beta_s', 'log10_A_s', 'log10_Ea_s'],
        'bounds': {
            'alpha_m': [0.5, 3.], 'beta_m': [0.5, 3.], 'log10_A_m': [3., 8.], 'log10_Ea_m': [4., 6.],
            'alpha_s': [0.5, 3.], 'beta_s': [0.5, 3.], 'log10_A_s': [2., 7.], 'log10_Ea_s': [4., 6.]
        }
    },
    'SERIES_VAR': {
        'names': ['alpha_1', 'beta_1', 'log10_A_1', 'log10_Ea_1',
                  'alpha_2', 'log10_A_2', 'log10_Ea_2'],
        'bounds': {
            'alpha_1': [0.5, 3.], 'beta_1': [0.5, 3.], 'log10_A_1': [3., 8.], 'log10_Ea_1': [4., 6.],
            'alpha_2': [0.5, 3.],                      'log10_A_2': [2., 7.], 'log10_Ea_2': [4., 6.]
        }
    },
    'AUTOCAT_VAR': {
        'names': ['alpha', 'beta', 'log10_A', 'log10_Ea'],
        'bounds': {
            'alpha': [0.5, 3.], 'beta': [0.5, 3.], # beta是自催化级数
            'log10_A': [3., 8.], 'log10_Ea': [4., 6.]
        }
    }
}

# 自动加载
W_BOUNDS_LOG = _BOUNDS_LIB[PHYSICS_MODEL_TYPE]['bounds']
CURRENT_PARAM_NAMES = _BOUNDS_LIB[PHYSICS_MODEL_TYPE]['names']

# 6. GP 参数
CONSTRAINT_NOISE_VAR     = [1e-6, 0.001]
CONSTRAINT_KERNEL_VAR    = [1e-3, 1.5]
CONSTRAINT_KERNEL_LEN    = 1e-2
PENALTY_LAMBDA = 0.0

# 7. 可视化
VIS_PLOT_DIAGNOSTIC = True
VIS_FONT = 'SimHei'

# 8. LHS 采样
SAMPLING_BOUNDS = {
    **W_BOUNDS_LOG,
    'k_var': CONSTRAINT_KERNEL_VAR,
    'l_1': [0.1, 2.0], 'l_2': [0.1, 2.0], 'l_3': [0.1, 2.0], 'l_4': [0.1, 2.0],
    'n_var': CONSTRAINT_NOISE_VAR
}