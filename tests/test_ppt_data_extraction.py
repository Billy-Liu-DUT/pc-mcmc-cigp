from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.extract_ppt_optimization_logs import extract_optimization_logs


PPT_PATH = Path(r"D:\工作二连续流\图\图改.pptx")


def test_extract_ppt_optimization_logs_writes_cigp_and_baseline_csvs():
    with TemporaryDirectory() as tmp:
        outputs = extract_optimization_logs(PPT_PATH, Path(tmp))

        assert outputs["cigp"].exists()
        assert outputs["standard_bo"].exists()
        assert "BO-15" in outputs["cigp"].read_text(encoding="utf-8")
        assert "BO-15" in outputs["standard_bo"].read_text(encoding="utf-8")
        assert "observed" in outputs["cigp"].read_text(encoding="utf-8").splitlines()[0]
