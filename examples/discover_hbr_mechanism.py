import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pc_mcmc_cigp.benchmarks import HBrMechanismBenchmark


def main() -> None:
    benchmark = HBrMechanismBenchmark(random_state=42, noise_level=0.02)
    result = benchmark.run_discovery(n_steps=500, burn_in=100)

    print("Posterior inclusion probabilities:")
    for reaction, pip in zip(benchmark.reactions, result.posterior_inclusion_probabilities):
        print(f"  {pip:6.2%}  {reaction.equation_str}")

    print("\nSelected reactions:")
    for reaction in result.selected_reactions:
        print(f"  {reaction.equation_str}")

    print(f"\nDiagnostics: {result.diagnostics}")


if __name__ == "__main__":
    main()
