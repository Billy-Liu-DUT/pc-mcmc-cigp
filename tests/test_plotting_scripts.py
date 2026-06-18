import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.plot_fig3 import plot_fig3
from scripts.plot_fig4 import plot_fig4, plot_fig4_comparison


def test_plot_fig3_writes_png_from_summary_json():
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        summary = {
            "benchmark": "hbr_mechanism_discovery",
            "reactions": [
                {"equation": "Br2 -> Br. + Br.", "pip": 0.91, "map_active": True},
                {"equation": "H2 + Br2 -> 2HBr", "pip": 0.03, "map_active": False},
            ],
        }
        summary_path = tmp_path / "summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

        output = plot_fig3(summary_path, tmp_path / "fig3.png")

        assert output.exists()
        assert output.stat().st_size > 0


def test_plot_fig4_writes_png_from_history_csv():
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        csv_path = tmp_path / "optimization_history.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["iteration", "yield", "best_yield", "x0", "x1", "x2", "x3"])
            writer.writeheader()
            writer.writerow({"iteration": 0, "yield": 0.1, "best_yield": 0.1, "x0": 1, "x1": 1, "x2": 320, "x3": 100})
            writer.writerow({"iteration": 1, "yield": 0.3, "best_yield": 0.3, "x0": 1, "x1": 1, "x2": 330, "x3": 200})

        output = plot_fig4(csv_path, tmp_path / "fig4.png")

        assert output.exists()
        assert output.stat().st_size > 0


def test_plot_fig4_comparison_writes_png_from_two_histories():
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cigp = tmp_path / "cigp.csv"
        baseline = tmp_path / "baseline.csv"
        for path, values in [(cigp, [0.2, 0.5]), (baseline, [0.1, 0.3])]:
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["iteration", "observed", "best"])
                writer.writeheader()
                for i, value in enumerate(values):
                    writer.writerow({"iteration": f"BO-{i + 1}", "observed": value, "best": max(values[: i + 1])})

        output = plot_fig4_comparison(cigp, baseline, tmp_path / "comparison.png")

        assert output.exists()
        assert output.stat().st_size > 0
