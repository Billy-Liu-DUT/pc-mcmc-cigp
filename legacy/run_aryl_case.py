# -*- coding: utf-8 -*-
import numpy as np
import config_discovery as cfg
from reaction_network_generator import AtomMappedNetworkGenerator, Species
from mechanism_engine import MechanismEngine
from mechanism_inference import SpikeAndSlabSampler


def generate_thermo_data(engine, t_span):
    """使用热力学参数生成真实数据"""
    # 设定真实参数 (G, Ea)
    # 假设: A+B -> I (DeltaG=-10, Ea=30), I+B -> P (DeltaG=-50, Ea=20)
    # 我们需要手动凑一下这些值对应的 k

    # 这里为了演示简单，我们还是反向操作：
    # 先设定我们想要的 k_true，然后让 data 生成
    # 只要 data 本身是物理的，MCMC 就能拟合出对应的 G 和 Ea

    # 真实机理: Step 1 (A+B->I), Step 2 (I+B->P)
    k_true = np.zeros(engine.n_reactions)
    z_true = np.zeros(engine.n_reactions)

    # 寻找并开启真实反应
    for i, r in enumerate(engine.reactions):
        s = r.equation_str
        if "A + B -> I_" in s:
            z_true[i] = 1.0;
            k_true[i] = 5.0  # 快
        if "I_" in s and "+ B -> P" in s:
            z_true[i] = 1.0;
            k_true[i] = 1.0  # 慢 (决速步)

    # 生成 LHS 数据
    dataset = []
    n_exp = cfg.DATA_SAMPLING_CONFIG['N_EXPERIMENTS']

    # 简单的浓度梯度
    for i in range(n_exp):
        y0 = np.zeros(engine.n_species)
        y0[engine.s_map['A']] = 1.0 + i * 0.2
        y0[engine.s_map['B']] = 2.0

        # 假设 T = 298K
        sim = engine.simulate(k_true, z_true, y0, t_span)
        if sim is None: continue

        # 加噪
        obs = sim + np.random.normal(0, 0.01, size=sim.shape)

        # 封装 (加入温度信息)
        obs_idx = [j for name, j in engine.s_map.items() if not name.startswith("I_")]
        dataset.append({
            't': t_span, 'y0_full': y0, 'T': 298.0,
            'data_matrix': obs[obs_idx, :], 'obs_indices': obs_idx
        })

    return dataset, k_true, z_true


def main():
    # 1. Init
    print(">>> 1. 构建物理空间 (Charge/Atom Mapped)...")
    # 这里会自动加载 config 里的 SPECIES_DEFINITIONS
    A = Species("A", cfg.SPECIES_DEFINITIONS['A'])
    B = Species("B", cfg.SPECIES_DEFINITIONS['B'])
    P = Species("P", cfg.SPECIES_DEFINITIONS['P'])  # P其实是 PhXY2

    gen = AtomMappedNetworkGenerator([A, B], [P])
    gen.generate_intermediates()
    reactions = gen.build_reaction_network()

    engine = MechanismEngine(gen.all_species, reactions)

    # 2. Data
    print(">>> 2. 生成热力学自洽的实验数据...")
    t_span = np.linspace(0, 5, 20)
    dataset, k_true, z_true = generate_thermo_data(engine, t_span)

    # 3. MCMC
    print(">>> 3. 启动 能量景观 MCMC 推断...")
    sampler = SpikeAndSlabSampler(engine, dataset)
    sampler.run()

    # 4. Report
    print(">>> 4. 生成分析报告...")
    sampler.plot_comprehensive_report(reactions)

    # 5. RDS Analysis (用 MCMC 得到的平均参数)
    print(">>> 5. 执行决速步 (RDS) 分析...")
    mean_theta = np.mean(sampler.chain_theta, axis=0)
    best_k = sampler._convert_theta_to_k(mean_theta, T=298.0)

    # 取概率 > 0.5 的反应结构
    pip = np.mean(sampler.chain_z, axis=0)
    best_z = (pip > 0.5).astype(float)

    # 用第一组实验条件做分析
    y0_test = dataset[0]['y0_full']
    engine.analyze_rds(best_k, best_z, y0_test, t_span)


if __name__ == "__main__":
    main()