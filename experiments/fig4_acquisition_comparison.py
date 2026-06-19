from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pc_mcmc_cigp.benchmarks import EpoxidationBenchmark


DEFAULT_STRATEGIES = ["PC_EI", "EI", "GWU", "DH", "UNCERTAINTY", "RANDOM"]


def run_experiment(
    output_dir: str | Path = "examples/outputs/fig4_acquisition_comparison",
    strategies: Iterable[str] = DEFAULT_STRATEGIES,
    seeds: Iterable[int] = range(10),
    n_initial: int = 8,
    n_iter: int = 15,
    n_candidates: int = 128,
    violation_threshold: float = 0.2,
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, float | int | str]] = []
    final_rows: list[dict[str, float | int | str]] = []
    for strategy in strategies:
        for seed in seeds:
            history = EpoxidationBenchmark(random_state=seed, noise_std=0.01).run_optimization(
                n_initial=n_initial,
                n_iter=n_iter,
                acquisition_name=strategy,
                n_candidates=n_candidates,
            )
            for row in history:
                raw_rows.append(_history_row(strategy, seed, row))
            bo_rows = [row for row in history if row["phase"] == "bo"]
            final_rows.append(
                {
                    "strategy": strategy,
                    "seed": seed,
                    "final_best": float(history[-1]["best_yield"]),
                    "bo_violations": sum(float(row["yield"]) < violation_threshold for row in bo_rows),
                }
            )

    raw_path = output_path / "raw_history.csv"
    summary_path = output_path / "summary.csv"
    _write_csv(
        raw_path,
        raw_rows,
        ["strategy", "seed", "iteration", "phase", "acquisition", "yield", "truth", "best_yield", "x0", "x1", "x2", "x3"],
    )
    _write_csv(summary_path, _summary_rows(final_rows), ["strategy", "n_runs", "mean_final_best", "std_final_best", "mean_bo_violations"])
    return raw_path, summary_path


def _history_row(strategy: str, seed: int, row: dict) -> dict[str, float | int | str]:
    x = row["x"]
    return {
        "strategy": strategy,
        "seed": seed,
        "iteration": row["iteration"],
        "phase": row["phase"],
        "acquisition": row["acquisition"],
        "yield": row["yield"],
        "truth": row["truth"],
        "best_yield": row["best_yield"],
        "x0": x[0],
        "x1": x[1],
        "x2": x[2],
        "x3": x[3],
    }


def _summary_rows(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    strategies = sorted({str(row["strategy"]) for row in rows})
    summary = []
    for strategy in strategies:
        subset = [row for row in rows if row["strategy"] == strategy]
        final_best = [float(row["final_best"]) for row in subset]
        bo_violations = [float(row["bo_violations"]) for row in subset]
        summary.append(
            {
                "strategy": strategy,
                "n_runs": len(subset),
                "mean_final_best": statistics.fmean(final_best),
                "std_final_best": statistics.stdev(final_best) if len(final_best) > 1 else 0.0,
                "mean_bo_violations": statistics.fmean(bo_violations),
            }
        )
    return summary


def _write_csv(path: Path, rows: list[dict[str, float | int | str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    raw_path, summary_path = run_experiment()
    print(f"Wrote {raw_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
