from __future__ import annotations

import csv
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
}

ROW_RE = re.compile(
    r"^(?P<iteration>(?:LHS|BO)-\d+)\s*\|\s*"
    r"(?P<styrene>[-+]?\d*\.?\d+)\s+"
    r"(?P<paa>[-+]?\d*\.?\d+)\s+"
    r"(?P<temperature>[-+]?\d*\.?\d+)\s+"
    r"(?P<time>[-+]?\d*\.?\d+)\s*\|\s*"
    r"(?P<pred>N/A|[-+]?\d*\.?\d+)\s+"
    r"(?P<observed>[-+]?\d*\.?\d+)\s+"
    r"(?P<truth>[-+]?\d*\.?\d+)\s*\|\s*"
    r"(?P<best>[-+]?\d*\.?\d+)"
)


def extract_optimization_logs(pptx_path: str | Path, output_dir: str | Path = "data") -> dict[str, Path]:
    pptx = Path(pptx_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    slide_texts = _slide_texts(pptx)
    cigp_rows = _extract_rows(_find_slide_text(slide_texts, "FINAL OPTIMIZATION REPORT", exclude="STANDARD BO"))
    baseline_rows = _extract_rows(_find_slide_text(slide_texts, "FINAL OPTIMIZATION REPORT (STANDARD BO"))

    outputs = {
        "cigp": output / "fig4_cigp.csv",
        "standard_bo": output / "fig4_standard_bo.csv",
    }
    _write_rows(outputs["cigp"], cigp_rows)
    _write_rows(outputs["standard_bo"], baseline_rows)
    return outputs


def _slide_texts(pptx: Path) -> list[str]:
    with zipfile.ZipFile(pptx) as archive:
        slide_names = sorted(
            [name for name in archive.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", name)],
            key=lambda value: int(re.search(r"slide(\d+)\.xml", value).group(1)),
        )
        texts = []
        for name in slide_names:
            root = ET.fromstring(archive.read(name))
            parts = [node.text for node in root.findall(".//a:t", NS) if node.text]
            texts.append("\n".join(parts))
        return texts


def _find_slide_text(slide_texts: list[str], include: str, exclude: str | None = None) -> str:
    for text in slide_texts:
        if include in text and (exclude is None or exclude not in text):
            return text
    raise ValueError(f"Could not find slide containing {include!r}")


def _extract_rows(text: str) -> list[dict[str, str | float]]:
    rows = []
    normalized = text.replace("\u00a0", " ")
    for raw_line in normalized.splitlines():
        line = " ".join(raw_line.split())
        match = ROW_RE.match(line)
        if not match:
            continue
        row = match.groupdict()
        rows.append(
            {
                "iteration": row["iteration"],
                "styrene": float(row["styrene"]),
                "paa": float(row["paa"]),
                "temperature": float(row["temperature"]),
                "time": float(row["time"]),
                "pred": "" if row["pred"] == "N/A" else float(row["pred"]),
                "observed": float(row["observed"]),
                "truth": float(row["truth"]),
                "best": float(row["best"]),
            }
        )
    if not rows:
        raise ValueError("No optimization rows parsed from PPT slide text")
    return rows


def _write_rows(path: Path, rows: list[dict[str, str | float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["iteration", "styrene", "paa", "temperature", "time", "pred", "observed", "truth", "best"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    pptx = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:\工作二连续流\图\图改.pptx")
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data")
    outputs = extract_optimization_logs(pptx, output_dir)
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
