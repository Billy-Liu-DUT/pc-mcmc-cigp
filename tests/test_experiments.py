import csv
import json
from tempfile import TemporaryDirectory
from pathlib import Path

from experiments.fig3_hbr_discovery import run_experiment as run_hbr_experiment
from experiments.fig4_epoxidation_bo import run_experiment as run_epoxidation_experiment


def test_fig3_experiment_writes_machine_readable_summary():
    with TemporaryDirectory() as tmp:
        summary_path = run_hbr_experiment(output_dir=Path(tmp), n_steps=30, burn_in=5, random_state=3)

        payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert payload["benchmark"] == "hbr_mechanism_discovery"
    assert payload["diagnostics"]["n_samples"] == 25
    assert payload["reactions"]


def test_fig4_experiment_writes_optimization_history_csv():
    with TemporaryDirectory() as tmp:
        csv_path = run_epoxidation_experiment(output_dir=Path(tmp), n_initial=3, n_iter=2, random_state=4)

        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

    assert len(rows) == 5
    assert {"iteration", "yield", "best_yield", "x0", "x1", "x2", "x3"}.issubset(rows[0])
