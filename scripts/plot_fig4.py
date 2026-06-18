from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_fig4(history_csv: str | Path, output_path: str | Path | None = None) -> Path:
    csv_path = Path(history_csv)
    rows = _read_rows(csv_path)
    iterations = [int(float(row["iteration"])) for row in rows]
    yields = [float(row["yield"]) for row in rows]
    best = [float(row["best_yield"]) for row in rows]

    output = Path(output_path) if output_path is not None else csv_path.with_name("fig4_optimization.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(iterations, best, color="#2F6B9A", linewidth=2.4, marker="o", label="Best so far")
    ax.scatter(iterations, yields, color="#D95F02", s=34, label="Observed yield", zorder=3)
    ax.set_xlabel("Experiment iteration")
    ax.set_ylabel("Yield")
    ax.set_title("CIGP-guided optimization trajectory")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.8)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def plot_fig4_comparison(
    cigp_csv: str | Path,
    baseline_csv: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    cigp_path = Path(cigp_csv)
    baseline_path = Path(baseline_csv)
    cigp = _read_rows(cigp_path)
    baseline = _read_rows(baseline_path)

    output = Path(output_path) if output_path is not None else cigp_path.with_name("fig4_cigp_vs_standard_bo.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    _plot_best_series(ax, cigp, "CIGP", "#2F6B9A")
    _plot_best_series(ax, baseline, "Standard BO", "#C94F3D")
    ax.set_xlabel("Experiment iteration")
    ax.set_ylabel("Best observed yield")
    ax.set_title("CIGP vs Standard BO optimization efficiency")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.8)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _iteration_number(label: str) -> int:
    if "-" in label:
        prefix, number = label.split("-", 1)
        base = 0 if prefix == "LHS" else 1000
        return base + int(number)
    return int(float(label))


def _plot_best_series(ax, rows: list[dict[str, str]], label: str, color: str) -> None:
    xs = list(range(1, len(rows) + 1))
    if "best" in rows[0]:
        best = [float(row["best"]) for row in rows]
    else:
        observed_key = "observed" if "observed" in rows[0] else "yield"
        values = [float(row[observed_key]) for row in rows]
        best = [max(values[: i + 1]) for i in range(len(values))]
    ax.plot(xs, best, marker="o", linewidth=2.2, color=color, label=label)


def main() -> None:
    default_cigp = Path("data/fig4_cigp.csv")
    default_baseline = Path("data/fig4_standard_bo.csv")
    if default_cigp.exists() and default_baseline.exists():
        output = plot_fig4_comparison(default_cigp, default_baseline)
    else:
        default_history = Path("examples/outputs/fig4_epoxidation/optimization_history.csv")
        output = plot_fig4(default_history)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
