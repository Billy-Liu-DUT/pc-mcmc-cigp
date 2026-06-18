from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_fig3(summary_json: str | Path, output_path: str | Path | None = None) -> Path:
    summary_path = Path(summary_json)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    reactions = payload["reactions"]
    labels = [row["equation"] for row in reactions]
    pips = [float(row["pip"]) for row in reactions]
    colors = ["#2F6B9A" if row.get("map_active") else "#B8B8B8" for row in reactions]

    output = Path(output_path) if output_path is not None else summary_path.with_name("fig3_pip.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, max(3.2, 0.42 * len(labels))))
    positions = range(len(labels))
    ax.barh(list(positions), pips, color=colors)
    ax.set_yticks(list(positions))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.axvline(0.5, color="#333333", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Posterior inclusion probability")
    ax.set_title("PC-MCMC reaction topology posterior")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def main() -> None:
    default_summary = Path("examples/outputs/fig3_hbr/summary.json")
    output = plot_fig3(default_summary)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
