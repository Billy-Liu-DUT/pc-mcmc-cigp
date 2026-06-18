# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import qmc
import config_discovery as cfg


class SpikeAndSlabSampler:
    def __init__(self, engine, dataset):
        self.engine = engine
        self.dataset = dataset
        self.n_rxns = engine.n_reactions
        self.n_species = engine.n_species

        # 确认模式
        self.use_thermo = cfg.MCMC_CONFIG.get('ENABLE_THERMO_CONSTRAINTS', False)
        self.isothermal = cfg.MCMC_CONFIG.get('ISOTHERMAL_MODE', False)  # <--- 获取新模式

        # 参数维度:
        if self.use_thermo:
            # 无论是全功能(G+Ea) 还是 等温(mu + lnk)，参数数量都是一样的
            # [0 : n_species] -> G 或 mu
            # [n_species : end] -> Ea 或 ln_k
            self.n_params = self.n_species + self.n_rxns
        else:
            self.n_params = self.n_rxns

        self.chain_z = []
        self.chain_theta = []
        self.chain_n_active = []

    def _convert_theta_to_k(self, theta, T=298.0):
        """将 MCMC 参数转化为 k 向量"""
        if not self.use_thermo:
            return theta

        # [新增] 等温模式处理
        if self.isothermal:
            # 解析 theta -> mu (无量纲势能), ln_k (对数速率)
            mu_vals = theta[:self.n_species]
            lnk_vals = theta[self.n_species:]

            # 构建 map
            mu_map = {s.name: mu_vals[i] for i, s in enumerate(self.engine.species)}

            # 调用新写的引擎函数
            k_vec = self.engine.calculate_isothermal_rates(mu_map, lnk_vals)
            return k_vec

        # (旧逻辑) 解析 theta -> G, Ea
        G_vals = theta[:self.n_species]
        Ea_vals = theta[self.n_species:]
        G_map = {s.name: G_vals[i] for i, s in enumerate(self.engine.species)}
        k_vec = self.engine.calculate_thermo_consistent_rates(G_map, Ea_vals, T)
        return k_vec

    def _lhs_initialization(self):
        sampler = qmc.LatinHypercube(d=self.n_params)
        sample = sampler.random(n=1)[0]

        if not self.use_thermo:
            lb, ub = cfg.MCMC_CONFIG['K_BOUNDS']
            return np.exp(np.log(lb) + sample * (np.log(ub) - np.log(lb)))

        else:
            theta = np.zeros(self.n_params)

            if self.isothermal:
                # === [新增] 等温模式初始化 ===
                # 1. mu (势能)
                mu_lb, mu_ub = cfg.MCMC_CONFIG['MU_BOUNDS']
                theta[:self.n_species] = mu_lb + sample[:self.n_species] * (mu_ub - mu_lb)
                # 2. ln_k (速率)
                lnk_lb, lnk_ub = cfg.MCMC_CONFIG['LN_K_BOUNDS']
                theta[self.n_species:] = lnk_lb + sample[self.n_species:] * (lnk_ub - lnk_lb)
            else:
                # (旧逻辑) G 和 Ea
                g_lb, g_ub = cfg.MCMC_CONFIG['G_BOUNDS']
                theta[:self.n_species] = g_lb + sample[:self.n_species] * (g_ub - g_lb)
                ea_lb, ea_ub = cfg.MCMC_CONFIG['Ea_BOUNDS']
                theta[self.n_species:] = ea_lb + sample[self.n_species:] * (ea_ub - ea_lb)

            return theta

    def _log_likelihood(self, theta, z):
        # ... (保持不变，因为 _convert_theta_to_k 已经处理了逻辑) ...
        # 注意：如果 dataset 里有 T，但在 isothermal 模式下，T 会被忽略，这是符合预期的
        # ...
        total_sse = 0
        loss_sigma = cfg.MCMC_CONFIG['SIGMA_LIKELIHOOD']

        for exp in self.dataset:
            T = exp.get('T', 298.0)
            k_at_T = self._convert_theta_to_k(theta, T)  # 这里会自动调用新逻辑

            exp_loss = self.engine.calculate_loss(k_at_T, z, [exp])
            total_sse += exp_loss

        if total_sse > 1e9: return -1e10
        return -0.5 * total_sse / (loss_sigma ** 2)

    def run(self):
        n_steps = cfg.MCMC_CONFIG['N_STEPS']
        burn_in = cfg.MCMC_CONFIG['BURN_IN']

        curr_theta = self._lhs_initialization()
        curr_z = np.zeros(self.n_rxns)
        # 随机开启
        if self.n_rxns > 0: curr_z[np.random.randint(0, self.n_rxns)] = 1.0

        curr_score = self._log_likelihood(curr_theta, curr_z)
        # Prior 简化: 稀疏先验 + 参数均匀先验(略)
        sparsity = cfg.MCMC_CONFIG['PRIOR_SPARSITY']

        pbar = tqdm(range(n_steps))
        for step in pbar:
            # 1. Update Parameters
            step_size = cfg.MCMC_CONFIG['STEP_SIZE_K']
            prop_theta = curr_theta + np.random.normal(0, step_size, size=self.n_params)

            # 简单边界检查 (Ea > 0)
            if self.use_thermo:
                prop_theta[self.n_species:] = np.abs(prop_theta[self.n_species:])

            prop_score = self._log_likelihood(prop_theta, curr_z)
            if np.log(np.random.rand()) < (prop_score - curr_score):
                curr_theta = prop_theta;
                curr_score = prop_score

            # 2. Update Structure
            flip = np.random.randint(0, self.n_rxns)
            prop_z = curr_z.copy();
            prop_z[flip] = 1.0 - prop_z[flip]

            # 稀疏惩罚项 change in log prior
            delta_n = np.sum(prop_z) - np.sum(curr_z)
            prior_diff = delta_n * (np.log(sparsity) - np.log(1 - sparsity))

            prop_score_z = self._log_likelihood(curr_theta, prop_z)

            if np.log(np.random.rand()) < (prop_score_z - curr_score + prior_diff):
                curr_z = prop_z;
                curr_score = prop_score_z

            if step >= burn_in:
                self.chain_z.append(curr_z.copy())
                self.chain_theta.append(curr_theta.copy())
                self.chain_n_active.append(np.sum(curr_z))

        return np.array(self.chain_z)

    def plot_comprehensive_report(self, reaction_list):
        """[核心报告] 生成图表 + 详细文字报告"""

        # 1. 计算统计量
        pip = np.mean(self.chain_z, axis=0)
        best_idx = np.argsort(pip)[::-1]

        # 获取最优参数 (均值)
        mean_theta = np.mean(self.chain_theta, axis=0)
        # 转化为 k (at 298K) 用于展示
        mean_k = self._convert_theta_to_k(mean_theta, T=298.0)

        print("\n" + "=" * 80)
        print("                AI 机理发现综合报告 (AI Mechanism Discovery Report)")
        print("=" * 80)
        print(f"{'Rank':<5} | {'PIP':<8} | {'k (298K)':<10} | {'Reaction Equation (Order Info)'}")
        print("-" * 80)

        top_reactions = []
        for i in range(min(10, len(reaction_list))):
            idx = best_idx[i]
            if pip[idx] < 0.05: break

            rxn = reaction_list[idx]
            # 分析级数
            order_info = []
            for r in rxn.reactants:
                order_info.append(f"Order({r.name})=1")  # 简化的基元反应级数
            order_str = ", ".join(order_info)

            print(f"{i + 1:<5} | {pip[idx]:.1%}   | {mean_k[idx]:.2e}   | {rxn.equation_str:<30} [{order_str}]")
            top_reactions.append(idx)

        print("=" * 80)

        # 2. 打印总包机理描述
        print("\n[AI 总结] 最可能的反应路径组合:")
        print("--------------------------------------------------")
        active_z = np.zeros(self.n_rxns)
        active_z[top_reactions] = 1.0  # 假设 Top PIP 的反应构成了机理

        for idx in top_reactions:
            print(f"  Step: {reaction_list[idx].equation_str}")

        # 3. 绘图 (保持之前的逻辑)
        self._plot_charts(reaction_list, pip)

    def _plot_charts(self, reaction_list, pip):
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            pass
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        # N分布
        n_counts = np.array(self.chain_n_active)
        u, c = np.unique(n_counts, return_counts=True)
        sns.barplot(x=u, y=c / len(n_counts), ax=ax[0], hue=u, legend=False, palette="Blues_d")
        ax[0].set_title("基元反应数量分布")

        # PIP分布
        top = np.argsort(pip)[::-1][:8]
        mask = pip[top] > 0.01
        if sum(mask) > 0:
            lbls = [reaction_list[i].equation_str for i in top[mask]]
            sns.barplot(x=pip[top][mask], y=lbls, ax=ax[1], hue=lbls, legend=False, palette="viridis")
            ax[1].set_title("反应路径概率 (PIP)")
        plt.tight_layout()
        plt.show()