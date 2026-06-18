from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pc_mcmc_cigp.benchmarks import EpoxidationBenchmark


def run_experiment(
    output_dir: str | Path = "examples/outputs/fig4_epoxidation",
    n_initial: int = 8,
    n_iter: int = 15,
    random_state: int = 42,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    history = EpoxidationBenchmark(random_state=random_state, noise_std=0.01).run_optimization(
        n_initial=n_initial,
        n_iter=n_iter,
    )
    csv_path = output_path / "optimization_history.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["iteration", "yield", "best_yield", "x0", "x1", "x2", "x3"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "iteration": row["iteration"],
                    "yield": row["yield"],
                    "best_yield": row["best_yield"],
                    "x0": row["x"][0],
                    "x1": row["x"][1],
                    "x2": row["x"][2],
                    "x3": row["x"][3],
                }
            )
    return csv_path


def main() -> None:
    csv_path = run_experiment()
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
