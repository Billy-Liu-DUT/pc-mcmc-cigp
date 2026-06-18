import numpy as np
from scipy.integrate import solve_ivp
import config_benchmark as cfg

R_GAS = 8.314

# --- 地面真值 (Ground Truth) 参数 ---
# 文献调研结果: Ea1=55kJ, Ea2=85kJ
GT_PARAMS = {
    'Ea1': 55000.0, 'logA1': 6.0,
    'Ea2': 85000.0, 'logA2': 10.0
}


def _gt_ode(t, y, k1, k2):
    S, PAA, E, Acid = y
    S, PAA, E, Acid = [max(0, x) for x in [S, PAA, E, Acid]]

    r1 = k1 * S * PAA  # 二级
    r2 = k2 * E * Acid  # 二级 (自毒化)

    return [-r1, -r1, r1 - r2, r1 - r2]


class EpoxyBenchmarkEnv:
    def __init__(self):
        self.bounds = cfg.DESIGN_SPACE['bounds']
        self.noise_std = 0.01  # 模拟实验误差

    def run_experiment(self, x_phys):
        """
        输入: x_phys [CA0, CB0, T, t]
        输出: yield (Scalar)
        """
        CA0, CB0, T, t_final = x_phys

        # 计算真值 k
        k1 = (10 ** GT_PARAMS['logA1']) * np.exp(-GT_PARAMS['Ea1'] / (R_GAS * T))
        k2 = (10 ** GT_PARAMS['logA2']) * np.exp(-GT_PARAMS['Ea2'] / (R_GAS * T))

        try:
            sol = solve_ivp(_gt_ode, [0, t_final], [CA0, CB0, 0.0, 0.0],
                            args=(k1, k2), method='LSODA', rtol=1e-6, atol=1e-9)
            if sol.success:
                y_true = sol.y[2, -1]
            else:
                y_true = 0.0
        except:
            y_true = 0.0

        # 添加噪声
        y_obs = max(0.0, y_true + np.random.normal(0, self.noise_std))
        return y_obs, y_true  # 返回观测值和真值(用于Regret计算)