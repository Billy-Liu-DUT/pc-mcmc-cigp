"""Optional SINDy baseline for the HBr benchmark.

Install optional dependencies first:

    pip install -e ".[benchmarks]"
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pc_mcmc_cigp.benchmarks import HBrMechanismBenchmark


def main() -> None:
    try:
        import pysindy as ps
    except ImportError as exc:
        raise SystemExit("pysindy is required for this baseline: pip install -e \".[benchmarks]\"") from exc

    benchmark = HBrMechanismBenchmark(random_state=42)
    dataset = benchmark.make_dataset(n_points=12)[0]
    x = dataset["data_matrix"].T
    t = dataset["t"]

    model = ps.SINDy()
    model.fit(x, t=t)
    print("SINDy baseline equations:")
    model.print()

    prediction = model.simulate(x[0], t)
    mse = float(np.mean((prediction - x) ** 2))
    print(f"Trajectory MSE: {mse:.6g}")


if __name__ == "__main__":
    main()
