from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pc_mcmc_cigp.reactions.network import Reaction, Species


@dataclass(frozen=True)
class ReactionPathway:
    reaction_indices: tuple[int, ...]
    species_sequence: tuple[str, ...]


class PathwayGenerator:
    """Enumerate bounded simple reaction paths between feed and target species."""

    def __init__(self, species: Sequence[Species], reactions: Sequence[Reaction]) -> None:
        self.species = tuple(species)
        self.reactions = tuple(reactions)

    def generate(self, sources: Sequence[str], targets: Sequence[str], max_steps: int = 5) -> list[ReactionPathway]:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        source_set, target_set = set(sources), set(targets)
        known_names = {s.name for s in self.species}
        if not source_set <= known_names or not target_set <= known_names:
            raise ValueError("sources and targets must be registered species")
        results: list[ReactionPathway] = []

        def visit(available: frozenset[str], used: tuple[int, ...], sequence: tuple[str, ...]) -> None:
            if target_set <= available and used:
                results.append(ReactionPathway(used, sequence))
                return
            if len(used) >= max_steps:
                return
            for idx, reaction in enumerate(self.reactions):
                if idx in used:
                    continue
                reactants = {s.name for s in reaction.reactants}
                products = {s.name for s in reaction.products}
                if reactants <= available and not products <= available:
                    visit(available | products, used + (idx,), sequence + tuple(sorted(products - available)))

        visit(frozenset(source_set), (), tuple(sorted(source_set)))
        unique = {p.reaction_indices: p for p in results}
        if not unique:
            return []
        shortest = min(len(indices) for indices in unique)
        return [path for indices, path in unique.items() if len(indices) == shortest]
