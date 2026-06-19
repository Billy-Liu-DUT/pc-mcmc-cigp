import csv
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.fig4_acquisition_comparison import run_experiment


def test_acquisition_comparison_writes_raw_history_and_summary():
    with TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        raw_path, summary_path = run_experiment(
            output_dir=output_dir,
            strategies=["PC_EI", "RANDOM"],
            seeds=[1, 2],
            n_initial=3,
            n_iter=1,
            n_candidates=16,
        )

        with raw_path.open(newline="", encoding="utf-8") as handle:
            raw_rows = list(csv.DictReader(handle))
        with summary_path.open(newline="", encoding="utf-8") as handle:
            summary_rows = list(csv.DictReader(handle))

    assert len(raw_rows) == 16
    assert {row["strategy"] for row in summary_rows} == {"PC_EI", "RANDOM"}
    assert {"strategy", "mean_final_best", "std_final_best", "mean_bo_violations"}.issubset(summary_rows[0])
