# -*- coding: utf-8 -*-
import numpy as np
import GPy
import sys
import traceback

# 导入你的项目模块
from series_reaction import SeriesReactionODE  # 使用连串反应作为测试物理模型
from cigp_model_normalized import CIGP_Normalized
from acquisition_functions import AcquisitionFactory


def run_test():
    print("=========================================================")
    print("   开始测试采集函数 (Acquisition Functions Smoke Test)")
    print("=========================================================")

    # 1. 准备伪造数据 (Mock Data)
    # 假设输入维度为 4 (CA0, CB0, T, t)，有 5 个观测点
    N_obs = 5
    D_input = 4

    # 生成 0-1 之间的随机归一化数据
    X_norm = np.random.rand(N_obs, D_input)
    Y_norm = np.random.rand(N_obs, 1)

    # 伪造 Scaler (归一化参数)
    X_scaler = {
        'min': np.array([0.0, 0.0, 300.0, 10.0]),
        'max': np.array([2.0, 2.0, 400.0, 100.0])
    }
    Y_scaler = {
        'mean': 0.5,
        'std': 0.2
    }

    print("[Step 1] 伪造数据准备完成。")

    # 2. 实例化物理模型和 CIGP 模型
    try:
        # 使用连串反应模型 (SeriesReactionODE)
        # 注意：这里 W_init 使用默认值
        physics_model = SeriesReactionODE(X_scaler=X_scaler)

        # 创建 RBF 核函数
        kernel = GPy.kern.RBF(input_dim=D_input, ARD=True)

        # 实例化 CIGP
        model = CIGP_Normalized(
            X_norm=X_norm,
            Y_norm=Y_norm,
            physics_model=physics_model,
            Y_scaler=Y_scaler,
            kernel=kernel
        )

        # 强制触发一次 parameters_changed 以确保内部矩阵 (如 woodbury_vector) 已计算
        model.parameters_changed()
        print(f"[Step 2] 模型实例化成功: {type(model).__name__}")

    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        traceback.print_exc()
        return

    # 3. 准备候选点 (Candidates)
    N_candidates = 10
    X_candidates = np.random.rand(N_candidates, D_input)
    print(f"[Step 3] 生成 {N_candidates} 个测试候选点。")

    # 4. 依次测试所有采集函数
    # 列表对应 AcquisitionFactory._REGISTRY 中的 Key
    strategies_to_test = ['EI', 'GWU', 'DH', 'PC_EI']

    print("\n--- 开始策略循环测试 ---")

    all_passed = True

    for strategy_name in strategies_to_test:
        print(f"\n正在测试策略: [{strategy_name}] ...")
        try:
            # 4.1 工厂实例化
            # 这里可以传入特定参数，例如 beta=2.0 或 threshold=0.1
            acq_func = AcquisitionFactory.get(strategy_name, beta=1.5, threshold=0.1)

            # 4.2 计算 Score
            # 传入 model, candidates, 和当前的 y_best (模拟值)
            scores = acq_func.compute(model, X_candidates, y_best=0.8)

            # 4.3 验证输出
            if scores.shape != (N_candidates,):
                raise ValueError(f"输出形状错误: 期望 ({N_candidates},), 实际 {scores.shape}")

            if not np.all(np.isfinite(scores)):
                print(f"⚠️ 警告: [{strategy_name}] 输出包含 NaN 或 Inf 值!")
                print("Scores:", scores)

            print(f"✅ PASS: [{strategy_name}] 运行正常。")
            print(f"   -> 样本得分示例: {scores[:3]}")  # 打印前3个分数看看量级

        except Exception as e:
            print(f"❌ FAILED: [{strategy_name}] 运行崩溃。")
            print(f"   错误信息: {e}")
            traceback.print_exc()
            all_passed = False

    print("\n=========================================================")
    if all_passed:
        print("🎉 测试结束: 所有采集函数均通过冒烟测试！")
    else:
        print("⚠️ 测试结束: 发现部分功能存在问题，请检查日志。")
    print("=========================================================")


if __name__ == "__main__":
    run_test()