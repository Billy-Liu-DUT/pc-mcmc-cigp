from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Species:
    """Chemical species represented by a name and atom/charge composition."""

    name: str
    atoms: dict[str, int] = field(default_factory=dict)

    def vector(self, atom_types: Sequence[str]) -> np.ndarray:
        return np.array([self.atoms.get(atom, 0) for atom in atom_types], dtype=float)


@dataclass(frozen=True)
class Reaction:
    """Elementary reaction candidate."""

    reactants: tuple[Species, ...] | list[Species]
    products: tuple[Species, ...] | list[Species]

    def __post_init__(self) -> None:
        object.__setattr__(self, "reactants", tuple(self.reactants))
        object.__setattr__(self, "products", tuple(self.products))

    @property
    def equation_str(self) -> str:
        lhs = " + ".join(species.name for species in self.reactants)
        rhs = " + ".join(species.name for species in self.products)
        return f"{lhs} -> {rhs}"

    def stoichiometric_vector(self, species_order: Sequence[Species]) -> np.ndarray:
        idx = {species.name: i for i, species in enumerate(species_order)}
        vec = np.zeros(len(species_order), dtype=float)
        for species in self.reactants:
            vec[idx[species.name]] -= 1.0
        for species in self.products:
            vec[idx[species.name]] += 1.0
        return vec

    def composition_delta(self, atom_types: Sequence[str]) -> np.ndarray:
        lhs = sum((species.vector(atom_types) for species in self.reactants), np.zeros(len(atom_types)))
        rhs = sum((species.vector(atom_types) for species in self.products), np.zeros(len(atom_types)))
        return rhs - lhs

    def is_balanced(self, atom_types: Sequence[str]) -> bool:
        return bool(np.allclose(self.composition_delta(atom_types), 0.0))


class AtomMappedNetworkGenerator:
    """Generate atom- and charge-balanced elementary reaction candidates."""

    def __init__(
        self,
        reactants: Sequence[Species],
        products: Sequence[Species],
        fragments: Sequence[Species] | None = None,
        max_reactants: int = 2,
        max_products: int = 2,
    ) -> None:
        self.reactants = list(reactants)
        self.products = list(products)
        self.fragments = list(fragments or [])
        self.max_reactants = max_reactants
        self.max_products = max_products
        self.species = self._unique_species([*self.reactants, *self.products, *self.fragments])
        atom_names = {atom for species in self.species for atom in species.atoms}
        self.atom_types = sorted(atom_names)
        if "Charge" not in self.atom_types:
            self.atom_types.append("Charge")

    def generate_intermediates(self) -> list[Species]:
        generated = list(self.species)
        for reactant in self.reactants:
            r_vec = reactant.vector(self.atom_types)
            for fragment in self.fragments:
                diff = r_vec - fragment.vector(self.atom_types)
                if self._valid_composition(diff):
                    atoms = {
                        atom: int(diff[i])
                        for i, atom in enumerate(self.atom_types)
                        if diff[i] != 0 or atom == "Charge"
                    }
                    generated.append(Species(f"I_{reactant.name}_loss_{fragment.name}", atoms))
        self.species = self._unique_species(generated)
        return self.species

    def generate(self) -> list[Reaction]:
        self.generate_intermediates()
        lhs_sets = self._side_combinations(self.max_reactants)
        rhs_sets = self._side_combinations(self.max_products)
        reactions: list[Reaction] = []
        seen: set[str] = set()

        for lhs in lhs_sets:
            lhs_names = sorted(species.name for species in lhs)
            lhs_vec = self._composition(lhs)
            for rhs in rhs_sets:
                rhs_names = sorted(species.name for species in rhs)
                if lhs_names == rhs_names:
                    continue
                if not np.allclose(lhs_vec, self._composition(rhs)):
                    continue
                reaction = Reaction(lhs, rhs)
                if reaction.equation_str not in seen:
                    seen.add(reaction.equation_str)
                    reactions.append(reaction)
        return reactions

    def _side_combinations(self, max_size: int) -> list[tuple[Species, ...]]:
        combos: list[tuple[Species, ...]] = []
        for size in range(1, max_size + 1):
            combos.extend(combinations_with_replacement(self.species, size))
        return combos

    def _composition(self, species: Iterable[Species]) -> np.ndarray:
        return sum((s.vector(self.atom_types) for s in species), np.zeros(len(self.atom_types)))

    def _valid_composition(self, vec: np.ndarray) -> bool:
        if np.allclose(vec, 0.0):
            return False
        for i, atom in enumerate(self.atom_types):
            if atom != "Charge" and vec[i] < 0:
                return False
        return True

    @staticmethod
    def _unique_species(species: Sequence[Species]) -> list[Species]:
        unique: dict[str, Species] = {}
        for item in species:
            unique[item.name] = item
        return list(unique.values())
