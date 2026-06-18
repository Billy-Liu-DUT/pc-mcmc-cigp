# -*- coding: utf-8 -*-
import numpy as np

# ==========================================
# 1. 物理/化学定义
# ==========================================
SPECIES_DEFINITIONS = {
    'H2': {'H': 2, 'Charge': 0},
    'Br2': {'Br': 2, 'Charge': 0},
    'HBr': {'H': 1, 'Br': 1, 'Charge': 0},
    'H.': {'H': 1, 'Charge': 0},
    'Br.': {'Br': 1, 'Charge': 0},
}

BASE_LEAVING_GROUPS = {
    'H.': {'H': 1, 'Charge': 0},
    'Br.': {'Br': 1, 'Charge': 0},
}

# ==========================================
# 2. 生成器设置
# ==========================================
GENERATOR_CONFIG = {
    'MAX_STOICH_INTERMEDIATE': 2,
    'MAX_REACTANTS': 2,
    'MAX_PRODUCTS': 2,
}

# ==========================================
# 3. 采样设置 (保持 Method of Isolation)
# ==========================================
DATA_SAMPLING_CONFIG = {
    'USE_LHS': True,
    'N_EXPERIMENTS': 8,
    'N_POINTS_PER_CURVE': 8,
    'TIME_SPAN': [0, 5.0],

    # [核心] 巨大的浓度差异是成功的关键
    'CONC_BOUNDS': {
        'H2': [0.5, 2.0],
        'Br2': [0.01, 10.0]
    },

    # [核心] 零噪声，逻辑验证专用
    'NOISE_LEVEL': 0.05,
    'TEMP_RANGE': [600, 600]
}

# ==========================================
# 4. MCMC 设置 (黄金平衡版)
# ==========================================
MCMC_CONFIG = {
    # --- 迭代 ---
    'N_STEPS': 10000,
    'BURN_IN': 2000,

    # [关键调整 1] 稀疏惩罚回调
    # 0.001 太松了，导致 AI 乱选 H2->2H。
    # 0.1 刚好，既能杀掉 H2->2H (Ghost)，又不至于杀掉真实机理。
    'PRIOR_SPARSITY': 0.005,

    # [关键调整 2] 极高精度
    # 0.05 太宽了。改回 0.01。
    # 只有真实机理 (0.5级) 能在 0.01 的误差下拟合 1000 倍的浓度跨度。
    # H2->2H (1级) 会因为误差超标被剔除。
    'SIGMA_LIKELIHOOD': 0.08,

    # --- 模式 ---
    'ENABLE_THERMO_CONSTRAINTS': True,
    'ISOTHERMAL_MODE': True,
    'R_GAS': 8.314e-3,
    'STIFFNESS_CAP_K': 1000.0,
    'ENABLE_DEACTIVATION': False,
    'DECAY_RATE': 0.0,

    # --- 范围 ---
    'MU_BOUNDS': [-3.0, 3.0],
    'LN_K_BOUNDS': [-1.0, 5.0],
    'STEP_SIZE_MCMC': 0.15,

    # 旧参数
    'G_BOUNDS': [-150.0, 50.0],
    'Ea_BOUNDS': [0.0, 100.0],
    'K_BOUNDS': [0.01, 100.0],
    'STEP_SIZE_K': 0.3,
}