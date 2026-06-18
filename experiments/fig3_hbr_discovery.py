from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pc_mcmc_cigp.benchmarks import HBrMechanismBenchmark


def run_experiment(
    output_dir: str | Path = "examples/outputs/fig3_hbr",
    n_steps: int = 2_000,
    burn_in: int = 500,
    random_state: int = 42,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    benchmark = HBrMechanismBenchmark(random_state=random_state, noise_level=0.02)
    result = benchmark.run_discovery(n_steps=n_steps, burn_in=burn_in)
    summary = {
        "benchmark": "hbr_mechanism_discovery",
        "random_state": random_state,
        "diagnostics": result.diagnostics,
        "reactions": [
            {
                "equation": reaction.equation_str,
                "pip": float(pip),
                "map_active": bool(active),
            }
            for reaction, pip, active in zip(
                benchmark.reactions,
                result.posterior_inclusion_probabilities,
                result.map_structure,
            )
        ],
        "selected_reactions": [reaction.equation_str for reaction in result.selected_reactions],
    }
    summary_path = output_path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def main() -> None:
    summary_path = run_experiment()
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
