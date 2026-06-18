# -*- coding: utf-8 -*-
import numpy as np
from itertools import combinations_with_replacement, product
from dataclasses import dataclass, field
from typing import Dict, List
import config_discovery as cfg


@dataclass(frozen=True)
class Species:
    name: str
    atoms: Dict[str, int] = field(default_factory=dict)

    def to_vector(self, atom_order: List[str]) -> np.ndarray:
        return np.array([self.atoms.get(a, 0) for a in atom_order])

    def __repr__(self):
        parts = []
        for k, v in self.atoms.items():
            if k == 'Charge': continue
            parts.append(f"{k}{v}" if v > 1 else f"{k}")
        charge = self.atoms.get('Charge', 0)
        q_str = f"({charge:+})" if charge != 0 else ""
        return f"{self.name}[{''.join(parts)}{q_str}]"


@dataclass
class Reaction:
    reactants: List[Species]
    products: List[Species]

    @property
    def equation_str(self) -> str:
        lhs = " + ".join([s.name for s in self.reactants])
        rhs = " + ".join([s.name for s in self.products])
        return f"{lhs} -> {rhs}"


class AtomMappedNetworkGenerator:
    def __init__(self, reactants: List[Species], products: List[Species]):
        self.reactants = reactants
        self.products = products
        self.base_species = reactants + products

        keys = set().union(*[s.atoms.keys() for s in self.base_species])
        if hasattr(cfg, 'BASE_LEAVING_GROUPS'):
            for frag_atoms in cfg.BASE_LEAVING_GROUPS.values():
                keys.update(frag_atoms.keys())
        self.atom_types = sorted(list(keys))
        if 'Charge' not in self.atom_types: self.atom_types.append('Charge')

        self.all_species = self.base_species.copy()
        self.fragments = []
        if hasattr(cfg, 'BASE_LEAVING_GROUPS'):
            for n, a in cfg.BASE_LEAVING_GROUPS.items():
                self.fragments.append(Species(n, a))

    def _check_is_valid_species(self, vec):
        for i, atom_name in enumerate(self.atom_types):
            if atom_name == 'Charge': continue
            if vec[i] < 0: return False
        if np.all(vec == 0): return False
        return True

    def _add_unique(self, new_species):
        new_vec = new_species.to_vector(self.atom_types)
        if not self._check_is_valid_species(new_vec): return None
        for s in self.all_species:
            if np.array_equal(s.to_vector(self.atom_types), new_vec): return s
        self.all_species.append(new_species)
        return new_species

    def generate_intermediates(self):
        print(f"\n[Step 1] 启动机理探索 (含电荷守恒)...")
        # 1. 解离
        for r in self.reactants:
            r_vec = r.to_vector(self.atom_types)
            for frag in self.fragments:
                diff_vec = r_vec - frag.to_vector(self.atom_types)
                name_hint = f"I_{r.name}_loss_{frag.name.replace('-', '').replace('+', '')}"
                new_atoms = {self.atom_types[k]: diff_vec[k] for k in range(len(self.atom_types))}
                added = self._add_unique(Species(name_hint, new_atoms))
                if added and added.name == name_hint:
                    self._add_unique(frag)  # 确保碎片也在全集中
                    print(f"    [发现解离] {r.name} -> {added} + {frag.name}")

        # 2. 结合
        current_pool = self.all_species.copy()
        pool_vecs = [s.to_vector(self.atom_types) for s in current_pool if s not in self.products]
        for i in range(len(pool_vecs)):
            for j in range(i, len(pool_vecs)):
                s1, s2 = current_pool[i], current_pool[j]
                if s1.atoms.get('Charge', 0) * s2.atoms.get('Charge', 0) > 0: continue

                vec_sum = pool_vecs[i] + pool_vecs[j]
                name_hint = f"I_{s1.name}_{s2.name}"
                new_atoms = {self.atom_types[k]: vec_sum[k] for k in range(len(self.atom_types))}
                res = self._add_unique(Species(name_hint, new_atoms))
                if res and res.name == name_hint:
                    print(f"    [发现结合] {s1.name} + {s2.name} -> {res}")

    def build_reaction_network(self):
        print(f"\n[Step 2] 构建反应网络 (Pool Size: {len(self.all_species)})...")
        valid_reactions = []
        pool = self.all_species
        max_r, max_p = cfg.GENERATOR_CONFIG['MAX_REACTANTS'], cfg.GENERATOR_CONFIG['MAX_PRODUCTS']

        lhs_list = []
        for r in range(1, max_r + 1): lhs_list.extend(combinations_with_replacement(pool, r))
        rhs_list = []
        for r in range(1, max_p + 1): rhs_list.extend(combinations_with_replacement(pool, r))

        for lhs in lhs_list:
            v_lhs = sum(s.to_vector(self.atom_types) for s in lhs)
            for rhs in rhs_list:
                v_rhs = sum(s.to_vector(self.atom_types) for s in rhs)
                if np.array_equal(v_lhs, v_rhs):
                    l_names, r_names = sorted([s.name for s in lhs]), sorted([s.name for s in rhs])
                    if l_names == r_names: continue
                    valid_reactions.append(Reaction(list(lhs), list(rhs)))

        print(f"  -> 穷举完成，共生成 {len(valid_reactions)} 个合法反应。")
        return valid_reactions