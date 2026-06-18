# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import norm
from abc import ABC, abstractmethod

"""
==================================================================================
CIGP 采集函数库 (Acquisition Function Library)
设计理念：融合物理机理 (Physics) 与数据驱动 (Data-Driven) 的混合策略
==================================================================================
"""


class BaseAcquisition(ABC):
    """采集函数策略基类"""

    def __init__(self, **kwargs):
        self.params = kwargs

    @abstractmethod
    def compute(self, model, X_candidates: np.ndarray, y_best: float = None) -> np.ndarray:
        """
        计算采集值 (Score)。
        :param model: CIGP_Normalized 模型实例
        :param X_candidates: 归一化的候选点矩阵 (N, D)
        :param y_best: 当前已知的最大观测值 (用于 EI)
        :return: scores (N, )
        """
        pass


# ------------------------------------------------------------------------------
# 1. 期望改进 (Expected Improvement)
# ------------------------------------------------------------------------------
class ExpectedImprovement(BaseAcquisition):
    """
    【工程基准】寻找全局最优产率，平衡探索与开发。
    """

    def compute(self, model, X_candidates, y_best=None):
        if y_best is None:
            # 如果未提供 y_best，默认使用预测均值的最大值
            mu_curr, _ = model.predict(model.X.values)
            y_best = np.max(mu_curr)

        mu, var = model.predict(X_candidates)
        sigma = np.sqrt(var)
        xi = self.params.get('xi', 0.01)  # 探索参数

        with np.errstate(divide='warn'):
            imp = mu - y_best - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma < 1e-9] = 0.0

        return ei.flatten()


# ------------------------------------------------------------------------------
# 2. 梯度加权不确定性 (Gradient-Weighted Uncertainty)
# ------------------------------------------------------------------------------
class GradientWeightedUncertainty(BaseAcquisition):
    """
    【科研导向】主动学习 (Active Learning) 以优化物理参数。
    原理：在模型不确定性高 且 对物理参数变化最敏感 的区域采样。
    公式: Score = sigma_total * || Jacobian_W ||
    """

    def compute(self, model, X_candidates, y_best=None):
        # 1. 预测总不确定性 (GP + Physics 修正后的不确定性)
        _, var = model.predict(X_candidates)
        std = np.sqrt(var)

        # 2. 计算物理模型对参数 W 的梯度 (Jacobian)
        # 调用 base_physics.py 中的方法
        # dmu_dW shape: (N_samples, N_params)
        dmu_dW = model.physics_model.compute_gradients_W(X_candidates, model.W.values)

        # 3. 计算梯度的 L2 范数 (表示综合敏感度)
        # norm shape: (N_samples, )
        sensitivity = np.linalg.norm(dmu_dW, axis=1)

        # 4. 组合
        score = std.flatten() * sensitivity
        return score


# ------------------------------------------------------------------------------
# 3. 结构性偏差捕获 (Discrepancy Hunter)
# ------------------------------------------------------------------------------
class DiscrepancyHunter(BaseAcquisition):
    """
    【CIGP 特色】寻找物理模型失效的区域。
    原理：专注于 g_err (GP 误差项) 大的区域，而非 Y 大的区域。
    公式: Score = |mu_err| + beta * sigma_err
    """

    def compute(self, model, X_candidates, y_best=None):
        # 1. 仅预测 GP 误差项 (g_err)
        mu_err, var_err = model.predict_g_err(X_candidates)
        std_err = np.sqrt(var_err)

        beta = self.params.get('beta', 1.0)

        # 2. 计算得分 (关注偏差的绝对值)
        # 我们想找到模型 "这种偏差" 最严重的地方，或者 "不知道有没有偏差" 的地方
        score = np.abs(mu_err) + beta * std_err

        return score.flatten()


# ------------------------------------------------------------------------------
# 4. 物理约束期望改进 (Physically-Constrained EI)
# ------------------------------------------------------------------------------
class PhysConstrainedEI(BaseAcquisition):
    """
    【安全/避坑】在物理模型认为合理的区域进行 EI 搜索。
    防止 GP 在物理上不可能的区域（如极低温、无反应物）产生虚假高值。
    """

    def compute(self, model, X_candidates, y_best=None):
        # 1. 计算基础 EI
        base_ei_strategy = ExpectedImprovement(**self.params)
        ei_score = base_ei_strategy.compute(model, X_candidates, y_best)

        # 2. 计算物理模型预测 (归一化空间)
        mu_phys = model.physics_model.compute_mean(X_candidates, model.W.values)

        # 3. 物理门控 (Soft Gating)
        # 设定一个软阈值，如果物理预测远低于 threshold，则权重 -> 0
        # 假设数据经过归一化，产率通常在 [0, 1] 或更高
        threshold = self.params.get('threshold', 0.1)  # 可调整
        sharpness = self.params.get('sharpness', 10.0)

        # Sigmoid 函数: 1 / (1 + exp(-k*(x-x0)))
        weight = 1.0 / (1.0 + np.exp(-sharpness * (mu_phys - threshold)))

        return ei_score * weight.flatten()


# ------------------------------------------------------------------------------
# 工厂模式 (Factory)
# ------------------------------------------------------------------------------
class AcquisitionFactory:
    _REGISTRY = {
        'EI': ExpectedImprovement,
        'GWU': GradientWeightedUncertainty,
        'DH': DiscrepancyHunter,
        'PC_EI': PhysConstrainedEI
    }

    @staticmethod
    def get(name: str, **kwargs) -> BaseAcquisition:
        if name not in AcquisitionFactory._REGISTRY:
            valid_keys = list(AcquisitionFactory._REGISTRY.keys())
            raise ValueError(f"未知的采集函数: '{name}'。可用选项: {valid_keys}")
        return AcquisitionFactory._REGISTRY[name](**kwargs)