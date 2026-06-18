import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pc_mcmc_cigp.benchmarks import EpoxidationBenchmark


def main() -> None:
    benchmark = EpoxidationBenchmark(random_state=42, noise_std=0.01)
    history = benchmark.run_optimization(n_initial=6, n_iter=8)

    print("Iter | Yield | Best | Conditions [Styrene, PAA, T, t]")
    print("-" * 72)
    for row in history:
        x = ", ".join(f"{v:.3g}" for v in row["x"])
        print(f"{row['iteration']:>4} | {row['yield']:.4f} | {row['best_yield']:.4f} | [{x}]")


if __name__ == "__main__":
    main()
