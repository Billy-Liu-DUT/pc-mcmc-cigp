from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
}


def inspect_pptx(pptx_path: str | Path, output_path: str | Path) -> Path:
    pptx = Path(pptx_path)
    output = Path(output_path)
    with zipfile.ZipFile(pptx) as archive:
        slides = sorted(
            [name for name in archive.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", name)],
            key=lambda value: int(re.search(r"slide(\d+)\.xml", value).group(1)),
        )
        lines = [
            "# PPT 源材料盘点",
            "",
            f"源文件：`{pptx}`",
            "",
            f"幻灯片数量：{len(slides)}",
            "",
        ]
        for idx, name in enumerate(slides, 1):
            root = ET.fromstring(archive.read(name))
            texts = [node.text.strip() for node in root.findall(".//a:t", NS) if node.text and node.text.strip()]
            pics = len(root.findall(".//p:pic", NS))
            graphic_frames = len(root.findall(".//p:graphicFrame", NS))
            shapes = len(root.findall(".//p:sp", NS))
            title = " / ".join(texts[:3])[:160] if texts else "(无明显文本)"
            lines.extend(
                [
                    f"## Slide {idx}",
                    f"- 文本摘要：{title}",
                    f"- 图片对象：{pics}",
                    f"- 图表/表格 frame：{graphic_frames}",
                    f"- 形状对象：{shapes}",
                ]
            )
            if texts:
                lines.append("- 全部文本：")
                lines.extend(f"  - {text}" for text in texts[:30])
            lines.append("")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python scripts/inspect_ppt_source.py <input.pptx> <output.md>")
    print(inspect_pptx(sys.argv[1], sys.argv[2]))


if __name__ == "__main__":
    main()
