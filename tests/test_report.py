import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_report_is_labeled_and_matches_the_markdown(tmp_path):
    # Every number in the README comes from this script, so the offline
    # path must run, label synthetic data as such, and write both outputs.
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_report.py"), "--synthetic-only",
         "--out-dir", str(tmp_path)],
        capture_output=True, text=True, check=True, cwd=ROOT,
    )  # fmt: skip
    md = (tmp_path / "results.md").read_text()
    html = (tmp_path / "report.html").read_text()
    assert "Synthetic data (generated in code, seed 42)" in md
    assert "Synthetic data (generated in code, seed 42)" in html
    assert "| Sharpe (rf = 0) |" in md and "Sharpe on the test window" in out.stdout
    assert "PEP" not in md  # offline mode never downloads
