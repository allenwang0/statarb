import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRICES = ROOT / "data" / "PEP_KO_2018-01-01_2024-12-31.csv"


def _run(tmp_path, *args):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_report.py"), *args,
         "--out-dir", str(tmp_path)],
        capture_output=True, text=True, check=True, cwd=ROOT,
    )  # fmt: skip


def test_synthetic_report_is_labeled_and_matches_the_markdown(tmp_path):
    # The offline path must run, label synthetic data as such, and write both outputs.
    out = _run(tmp_path, "--synthetic-only")
    md = (tmp_path / "results.md").read_text()
    html = (tmp_path / "index.html").read_text()
    assert "Synthetic data (generated in code, seed 42)" in md
    assert "Synthetic data (generated in code, seed 42)" in html
    assert "| Sharpe (rf = 0) |" in md and "Sharpe on the test window" in out.stdout
    assert "PEP" not in md  # offline mode never downloads


def test_published_report_leads_with_the_real_result_and_its_caveats(tmp_path):
    # CI builds Pages from the committed prices. A skimmer must meet the
    # real-data loss and the reasons not to trust either hedge (no
    # cointegration, misfit Kalman model) before the synthetic sanity check.
    _run(tmp_path, "--prices", str(PRICES.relative_to(ROOT)))
    html = (tmp_path / "index.html").read_text()
    head = html[: html.index('id="oos"')]
    assert "the Kalman hedge lost to a plain rolling hedge" in head
    assert "−0.44" in head and "0.62" in head
    assert "p = 0.182" in head and "fails at 0.05" in head
    assert "20.17" in head
    assert html.index('id="sanity"') > html.index('id="fit"')
    assert "100 % hit rate comes from only 5 trades" in html
    assert 'name="viewport"' in html and "prefers-color-scheme: dark" in html
    assert "plotly" not in html.lower()
