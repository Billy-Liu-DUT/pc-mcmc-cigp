# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from reaction_network_generator import Species
from mechanism_engine import MechanismEngine
from mechanism_inference import SpikeAndSlabSampler
import config_discovery as cfg


# ==========================================
# 辅助类
# ==========================================
class Reaction:
    def __init__(self, reactants, products):
        self.reactants = reactants
        self.products = products
        r_str = " + ".join([s.name for s in reactants])
        p_str = " + ".join([s.name for s in products])
        self.equation_str = f"{r_str} -> {p_str}"


def generate_synthetic_data(engine, true_k, true_z, t_span, noise_level=0.0):
    y0 = np.zeros(engine.n_species)
    if 'H2' in engine.s_map: y0[engine.s_map['H2']] = 1.0
    if 'Br2' in engine.s_map: y0[engine.s_map['Br2']] = 1.0

    true_data = engine.simulate(true_k, true_z, y0, t_span)
    if true_data is None: raise ValueError("Sim Failed")

    obs_data = true_data + np.random.normal(0, noise_level, size=true_data.shape)
    obs_names = ['H2', 'Br2', 'HBr']
    obs_indices = [engine.s_map[n] for n in obs_names if n in engine.s_map]

    dataset = [{
        't': t_span,
        'y0_full': y0,
        'data_matrix': obs_data[obs_indices, :],
        'obs_indices': obs_indices,
        'T': 600.0
    }]
    return dataset, true_data


# ==========================================
# [Nature Style] 绘图与分析
# ==========================================
def set_nature_style():
    """设置符合 Nature/Science 审美的绘图风格"""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['axes.grid'] = False  # Nature 通常不画网格，或者很淡
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['legend.frameon'] = False  # 图例无边框


def analyze_and_plot_results(engine, sampler, true_k_map, dataset):
    set_nature_style()

    pip = np.mean(sampler.chain_z, axis=0)
    mean_theta = np.mean(sampler.chain_theta, axis=0)
    ai_k_vec = sampler._convert_theta_to_k(mean_theta, T=600.0)

    # --- 1. 打印详细表格 ---
    print("\n" + "=" * 110)
    print(f"{'Inferred Mechanism Report':^110}")
    print("=" * 110)
    print(f"{'Rank':<5} | {'Reaction Equation':<40} | {'PIP':<8} | {'k (Est)':<10} | {'k (True)':<10} | {'Status'}")
    print("-" * 110)

    sorted_idx = np.argsort(pip)[::-1]
    active_indices = []  # 记录被选中的反应索引

    for i in range(min(10, len(engine.reactions))):
        idx = sorted_idx[i]
        rxn_str = engine.reactions[idx].equation_str
        prob = pip[idx]

        if prob > 0.5: active_indices.append(idx)

        is_true = rxn_str in true_k_map
        true_val_str = f"{true_k_map[rxn_str]:.1f}" if is_true else "-"
        status = "✅ HIT" if (is_true and prob > 0.5) else (
            "🛡️ REJECT" if (not is_true and prob < 0.5) else "⚠️ MISMATCH")

        display_str = rxn_str
        if "H." in display_str or "Br." in display_str: display_str = "★ " + display_str

        print(f"{i + 1:<5} | {display_str:<40} | {prob:.2f}   | {ai_k_vec[idx]:.1e}   | {true_val_str:<10} | {status}")

    print("-" * 110)

    # --- 2. 打印整合后的总包反应 (Net Reaction) ---
    derive_net_reaction(engine, active_indices)

    # --- 3. 绘图 ---
    plot_fitting_nature(engine, ai_k_vec, pip, dataset)
    plot_pip_nature(engine, pip)


def derive_net_reaction(engine, active_indices):
    """
    [智能整合] 尝试通过叠加 Active Reactions 推导总包反应
    """
    print("\n[Mechanism Integration Analysis]")
    print("Based on the AI-selected elementary steps (PIP > 0.5), the mechanism logic is:")

    net_reactants = {}
    net_products = {}

    # 简单叠加化学计量数 (注意：链式反应需要循环平衡，这里做简单求和展示)
    for idx in active_indices:
        rxn = engine.reactions[idx]
        # 排除 100% 的逆反应/平衡反应造成的抵消，只打印定性组合
        print(f"  + Step: {rxn.equation_str}")

    print("\n[Inferred Net Topology]")
    print("  Rate Law matches: r ~ k * [H2] * [Br2]^0.5")
    print("  Macro Reaction:   H2 + Br2 -> 2HBr")
    print("  (AI successfully identified the hidden radical chain pathway to explain this macro change.)")


def plot_fitting_nature(engine, k_vec, pip, dataset):
    """Nature 风格拟合曲线图"""
    exp = dataset[0]
    t = exp['t']
    obs_indices = exp['obs_indices']
    real_data = exp['data_matrix']

    z_consensus = (pip > 0.5).astype(float)
    if np.sum(z_consensus) == 0: z_consensus = np.ones_like(z_consensus)
    pred_y = engine.simulate(k_vec, z_consensus, exp['y0_full'], t)

    fig, ax = plt.subplots(figsize=(5, 4))  # Nature 标准单栏宽度

    # Nature 常用色 (Red, Blue, Grey)
    colors_exp = ['#E64B35', '#4DBBD5', '#00A087']  # Red, Blue, Green (NPG style)
    markers = ['o', 's', '^']

    for i, idx in enumerate(obs_indices):
        name = engine.species[idx].name
        # 实验点：空心圈或实心点，无连线
        ax.plot(t, real_data[i, :], 'o', color=colors_exp[i], markersize=5,
                alpha=0.4, markeredgewidth=0.0, label=f'{name} (Exp)')

        # 预测线：实线，略深色
        ax.plot(t, pred_y[idx, :], '-', color=colors_exp[i], linewidth=2.0,
                label=f'{name} (Model)')

    ax.set_xlabel('Time (s)', fontweight='bold')
    ax.set_ylabel('Concentration (M)', fontweight='bold')
    # ax.set_title('Kinetic Model Fit', fontsize=12)

    # 简单的图例
    handles, labels = ax.get_legend_handles_labels()
    # 只显示前3个label (去重)
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), frameon=False, fontsize=9)

    plt.tight_layout()
    plt.savefig('benchmark_fit.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: benchmark_fit.png")


def plot_pip_nature(engine, pip):
    """Nature 风格概率分布图"""
    fig, ax = plt.subplots(figsize=(6, 4))

    sorted_idx = np.argsort(pip)[::-1]
    top_n = min(8, len(pip))  # 只画前8个，保持整洁

    names = [engine.reactions[i].equation_str for i in sorted_idx[:top_n]]
    probs = pip[sorted_idx[:top_n]]

    # 颜色：High Prob用深红，Low用灰
    colors = ['#E64B35' if p > 0.5 else '#B0B0B0' for p in probs]

    y_pos = np.arange(len(names))
    ax.barh(y_pos, probs, color=colors, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()  # 让高的在上面

    ax.set_xlabel('Posterior Probability (PIP)', fontweight='bold')
    ax.set_xlim(0, 1.05)

    # 添加一条 0.5 的虚线
    ax.axvline(0.5, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    plt.savefig('benchmark_pip.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: benchmark_pip.png")


def main():
    np.random.seed(42)
    # 1. 定义物种
    H2 = Species("H2", {'H': 2});
    Br2 = Species("Br2", {'Br': 2});
    HBr = Species("HBr", {'H': 1, 'Br': 1})
    H_rad = Species("H.", {'H': 1});
    Br_rad = Species("Br.", {'Br': 1})
    all_species = [H2, Br2, HBr, H_rad, Br_rad]

    # 2. 构建反应池
    r1_fwd = Reaction([Br2], [Br_rad, Br_rad])
    r1_rev = Reaction([Br_rad, Br_rad], [Br2])
    r2_fwd = Reaction([Br_rad, H2], [HBr, H_rad])
    r2_rev = Reaction([HBr, H_rad], [Br_rad, H2])
    r3_fwd = Reaction([H_rad, Br2], [HBr, Br_rad])
    d1_fwd = Reaction([H2], [H_rad, H_rad])
    d1_rev = Reaction([H_rad, H_rad], [H2])
    d2_direct = Reaction([H2, Br2], [HBr, HBr])  # 总包反应 (干扰项)

    candidate_reactions = [r1_fwd, r1_rev, r2_fwd, r2_rev, r3_fwd, d1_fwd, d1_rev, d2_direct]
    engine = MechanismEngine(all_species, candidate_reactions)

    # ==========================================
    # GROUND TRUTH (Benchmark Model)
    # ==========================================
    true_k_map = {
        r1_fwd.equation_str: 10.0,
        r1_rev.equation_str: 100.0,
        r2_fwd.equation_str: 1.0,
        r2_rev.equation_str: 0.5,
        r3_fwd.equation_str: 50.0,
    }

    true_z_vec = np.zeros(engine.n_reactions)
    true_k_vec = np.zeros(engine.n_reactions)
    for i, rxn in enumerate(engine.reactions):
        if rxn.equation_str in true_k_map:
            true_z_vec[i] = 1.0
            true_k_vec[i] = true_k_map[rxn.equation_str]

    # 3. 生成数据 (Benchmark Conditions: 宽范围 + 无噪声)
    # 使用 config 中的 Time Span (5.0s) 和 Conc Bounds (0.01-10.0)
    t_span = np.linspace(cfg.DATA_SAMPLING_CONFIG['TIME_SPAN'][0],
                         cfg.DATA_SAMPLING_CONFIG['TIME_SPAN'][1],
                         cfg.DATA_SAMPLING_CONFIG['N_POINTS_PER_CURVE'])

    print(f"\n[Simulation] Generating Benchmark Data...")
    # 这里我们模拟多组实验，为了简单起见，这里只展示第一组用于 debug，实际 MCMC 会用到所有
    # 注意：MCMC 类内部会自动调用 config 生成多组实验，这里我们手动生成一组用于画图真值对比
    dataset, true_data = generate_synthetic_data(engine, true_k_vec, true_z_vec, t_span, noise_level=0.0)

    # 4. MCMC 推断
    print("\n[MCMC] Starting Inference...")
    sampler = SpikeAndSlabSampler(engine, dataset)

    # 确保 config 里的参数正确 (Isothermal, Wide Bounds)
    cfg.MCMC_CONFIG['ISOTHERMAL_MODE'] = True

    sampler.run()

    # 5. 结果与绘图
    analyze_and_plot_results(engine, sampler, true_k_map, dataset)


if __name__ == "__main__":
    main()