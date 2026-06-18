# -*- coding: utf-8 -*-
import numpy as np
from scipy.integrate import solve_ivp
import config_discovery as cfg


class MechanismEngine:
    def __init__(self, all_species, all_reactions):
        self.species = all_species
        self.reactions = all_reactions
        self.n_species = len(all_species)
        self.n_reactions = len(all_reactions)
        self.s_map = {s.name: i for i, s in enumerate(self.species)}

        # 构建化学计量数矩阵 S
        self.S = np.zeros((self.n_species, self.n_reactions))
        for j, rxn in enumerate(self.reactions):
            for r in rxn.reactants: self.S[self.s_map[r.name], j] -= 1.0
            for p in rxn.products: self.S[self.s_map[p.name], j] += 1.0

        # 预计算反应级数模式，加速 ODE
        self.rate_patterns = []
        for rxn in self.reactions:
            reactants_idx = [self.s_map[r.name] for r in rxn.reactants]
            counts = {}
            for idx in reactants_idx: counts[idx] = counts.get(idx, 0) + 1
            self.rate_patterns.append(list(counts.items()))

        # ==========================================
        # [FIX START] 1. 自动识别可逆反应对
        # ==========================================
        # 目的：建立逆反应索引 -> 正反应索引的映射
        # 只要建立了这个映射，我们在计算 k 时就能强制约束 Ea
        self.reverse_map = {}  # key: reverse_idx, value: primary_idx
        self._detect_reversible_pairs()
        # ==========================================

    def _detect_reversible_pairs(self):
        """[内部辅助] 扫描反应列表，找出互为逆反应的对子"""

        # 辅助函数：将反应物/产物转化为集合以便比较
        def get_species_sets(rxn):
            return frozenset(s.name for s in rxn.reactants), frozenset(s.name for s in rxn.products)

        rxn_sigs = [get_species_sets(r) for r in self.reactions]

        for i in range(self.n_reactions):
            # 如果 i 已经被标记为别人的逆反应，跳过，避免重复
            if i in self.reverse_map: continue

            r_i, p_i = rxn_sigs[i]

            # 向后寻找 i 的逆反应 j
            for j in range(i + 1, self.n_reactions):
                if j in self.reverse_map: continue  # 已经被配对过了

                r_j, p_j = rxn_sigs[j]

                # 判断标准：i 的反应物 == j 的产物 且 i 的产物 == j 的反应物
                if r_i == p_j and p_i == r_j:
                    self.reverse_map[j] = i  # j 依赖于 i
                    # print(f"[Debug] Linked Pair: Rxn {i} (Primary) <-> Rxn {j} (Reverse)")
                    # 一个反应只配对一次，找到后 break
                    break

    def calculate_thermo_consistent_rates(self, G_map, Ea_map, T):
        """
        [热力学核心 - 修复版]
        从能量 G 和活化能 Ea 推导 k，并严格执行详细平衡约束。
        """
        R = cfg.MCMC_CONFIG['R_GAS']
        k_vector = np.zeros(self.n_reactions)

        # 用于缓存计算后实际使用的 Ea，供逆反应调用
        used_Ea_cache = np.zeros(self.n_reactions)

        # 假设 A 因子为常量 (10^10)，实际也可采样
        A_factor = 1e10

        for i, rxn in enumerate(self.reactions):
            # 1. 计算 Delta G (基于物种能量采样，天然满足循环一致性)
            G_r = sum([G_map.get(s.name, 0.0) for s in rxn.reactants])
            G_p = sum([G_map.get(s.name, 0.0) for s in rxn.products])
            delta_G = G_p - G_r

            # ==========================================
            # [FIX START] 2. 区分主反应与逆反应，强制 Ea 约束
            # ==========================================
            if i in self.reverse_map:
                # Case A: 这是一个逆反应 (Dependent)
                primary_idx = self.reverse_map[i]

                # 获取对应主反应实际使用的 Ea
                Ea_primary = used_Ea_cache[primary_idx]

                # 物理约束推导：
                # Ea_rev = Ea_fwd - DeltaG_fwd
                # 因为 DeltaG_rev (即当前的 delta_G) = - DeltaG_fwd
                # 所以 Ea_rev = Ea_fwd + DeltaG_rev
                Ea_actual = Ea_primary + delta_G

                # 注意：这里完全忽略了 MCMC 为反应 i 采样的 Ea_map[i]
                # 从而消除了独立采样导致的物理违背

            else:
                # Case B: 这是一个主反应 (Primary) 或不可逆反应
                Ea_tentative = Ea_map[i]

                # [物理约束] 吸热反应能垒必须 > Delta G
                # 如果要爬升 50kJ (delta_G > 0)，那能垒至少得 > 50
                if delta_G > 0 and Ea_tentative < delta_G:
                    Ea_actual = delta_G + 1.0  # 强制修正
                else:
                    Ea_actual = Ea_tentative

            # 缓存实际使用的 Ea
            used_Ea_cache[i] = Ea_actual
            # ==========================================

            # 3. 计算 k (Arrhenius)
            # 防止 Ea 为负 (理论上经过上述逻辑不应出现，但在极端数值下为了安全)
            if Ea_actual < 0: Ea_actual = 0.0

            k_val = A_factor * np.exp(-Ea_actual / (R * T))

            # [数值保护] 刚性/快平衡截断
            cap = cfg.MCMC_CONFIG.get('STIFFNESS_CAP_K', 1e5)
            if k_val > cap: k_val = cap

            k_vector[i] = k_val

        return k_vector

    def _ode_func(self, t, y, k_vector, z_structure):
        C = np.maximum(y, 0.0)
        r = np.zeros(self.n_reactions)

        # [失活] 催化剂失活因子
        decay_factor = 1.0
        if cfg.MCMC_CONFIG.get('ENABLE_DEACTIVATION', False):
            kd = cfg.MCMC_CONFIG.get('DECAY_RATE', 0.0)
            decay_factor = np.exp(-kd * t)

        for j in range(self.n_reactions):
            if z_structure[j] < 0.5: continue

            # k * decay
            val = k_vector[j] * decay_factor

            for (s_idx, power) in self.rate_patterns[j]:
                val *= (C[s_idx] ** power)
            r[j] = val

        return self.S @ r

    def simulate(self, k_vector, z_structure, y0, t_eval):
        sol = solve_ivp(
            fun=lambda t, y: self._ode_func(t, y, k_vector, z_structure),
            t_span=(t_eval[0], t_eval[-1]),
            y0=y0, t_eval=t_eval, method='LSODA', rtol=1e-5, atol=1e-7
        )
        return sol.y if sol.success else None

    def calculate_loss(self, k_vector, z_structure, dataset):
        total_error = 0.0
        for exp in dataset:
            t = exp['t']
            y0_full = exp['y0_full']

            # 如果启用了热力学，这里的 k_vector 是已经针对该温度 T 计算好的
            # 如果没启用，k_vector 就是纯参数

            pred_y = self.simulate(k_vector, z_structure, y0_full, t)
            if pred_y is None: return 1e10

            obs_indices = exp['obs_indices']
            pred_obs = pred_y[obs_indices, :]
            real_obs = exp['data_matrix']

            sq_diff = (pred_obs - real_obs) ** 2
            total_error += np.sum(sq_diff)
        return total_error

    def analyze_rds(self, k_vector, z_structure, y0, t_eval):
        """[化学洞察] 决速步分析 (DRC)"""
        print("\n========= 决速步 (RDS) 智能识别 =========")
        active_idx = np.where(z_structure > 0.5)[0]
        if len(active_idx) == 0: return

        # 1. 基准
        sol_base = self.simulate(k_vector, z_structure, y0, t_eval)

        # 寻找目标产物 P 的索引
        p_idx = -1
        if 'P' in self.s_map:
            p_idx = self.s_map['P']
        else:
            # 备选：如果没有 P，取最后一个物种
            p_idx = self.n_species - 1

        y_base = sol_base[p_idx, -1]

        sensitivity = {}
        for idx in active_idx:
            k_new = k_vector.copy()
            k_new[idx] *= 1.01  # +1%
            sol_new = self.simulate(k_new, z_structure, y0, t_eval)
            y_new = sol_new[p_idx, -1]
            # Sens = (% dY) / (% dk)
            # 防止除以 0
            denom = y_base if abs(y_base) > 1e-9 else 1e-9
            sens = ((y_new - y_base) / denom) / 0.01
            sensitivity[idx] = sens

        total = sum(abs(v) for v in sensitivity.values()) + 1e-9

        print(f"{'Reaction':<30} | {'DRC Coeff':<10} | {'Role'}")
        print("-" * 60)
        for idx, val in sensitivity.items():
            norm = abs(val) / total
            role = "★ RDS" if norm > 0.5 else ("☆ Key" if norm > 0.1 else "-")
            print(f"{self.reactions[idx].equation_str:<30} | {norm:.2%}     | {role}")
        print("=" * 60)

    def calculate_isothermal_rates(self, mu_map, ln_k_fwd_map):
        """
        [等温势能节点法]
        直接从无量纲势能 (mu) 和 正向速率对数 (ln_k) 计算自洽的 k 向量。
        """
        k_vector = np.zeros(self.n_reactions)
        used_k_fwd_cache = np.zeros(self.n_reactions)  # 缓存主反应的 k_fwd

        for i, rxn in enumerate(self.reactions):
            # 1. 计算反应的势能变 Delta mu (相当于 Delta G / RT)
            # Delta mu = sum(mu_products) - sum(mu_reactants)
            mu_r = sum([mu_map.get(s.name, 0.0) for s in rxn.reactants])
            mu_p = sum([mu_map.get(s.name, 0.0) for s in rxn.products])
            delta_mu = mu_p - mu_r

            # K_eq = exp(-Delta G / RT) = exp(-Delta mu)
            # 为了防止数值溢出，做个截断
            if delta_mu > 100:
                K_eq = 1e-43
            elif delta_mu < -100:
                K_eq = 1e43
            else:
                K_eq = np.exp(-delta_mu)

            # 2. 区分主反应和逆反应
            if i in self.reverse_map:
                # === 逆反应 (Dependent) ===
                primary_idx = self.reverse_map[i]

                # 获取对应主反应的 k_fwd
                k_forward_primary = used_k_fwd_cache[primary_idx]

                # 物理约束: k_rev = k_fwd / K_eq_primary
                # 注意：当前反应是逆反应，它的 K_eq 是主反应 K_eq 的倒数
                # 即: K_eq_this = 1 / K_eq_primary
                # k_this (rev) = k_primary * K_eq_this_reaction (详细平衡)
                # 或者更简单的推导：
                # A -> B (k1), B -> A (k2). K = k1/k2 = exp(-(mu_B - mu_A))
                # k2 = k1 * exp(mu_B - mu_A) = k1 * exp(delta_mu_primary)

                # 由于我们上面算的是当前反应的 delta_mu (即 mu_A - mu_B)，
                # 所以 k_this = k_primary * exp(- delta_mu_this)
                # 等等，直接用 detailed balance:
                # Rate_net = k_fwd[A] - k_rev[B] = 0 => k_fwd/k_rev = [B]/[A] = K_eq
                # 所以 k_rev = k_fwd / K_eq_primary

                # 让我们用最稳妥的方式：
                # 当前是逆反应，那么它的 k = 主反应k / 主反应Keq
                # 主反应 Keq = exp(-(mu_p_primary - mu_r_primary))
                #            = exp(-(mu_r_current - mu_p_current))
                #            = exp(delta_mu_current)

                k_val = k_forward_primary * np.exp(delta_mu)

            else:
                # === 主反应 (Primary) ===
                # 直接使用采样的 ln_k
                ln_k = ln_k_fwd_map[i]
                k_val = np.exp(ln_k)

                # 缓存它供逆反应使用
                used_k_fwd_cache[i] = k_val

            # [数值保护]
            cap = cfg.MCMC_CONFIG.get('STIFFNESS_CAP_K', 1e6)
            if k_val > cap: k_val = cap
            if k_val < 1e-10: k_val = 1e-10

            k_vector[i] = k_val

        return k_vector