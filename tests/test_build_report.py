"""The published case study must say what results/case_study.json says.

Expected values are recomputed here from the JSON, independently of
scripts/build_report.py, so a generator bug (wrong lag, wrong model, train
instead of test window, a negative number promoted to the hero) fails a test
instead of reaching the page.
"""

import html
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = json.loads((ROOT / "results" / "case_study.json").read_text())
MINUS = "−"


def sh(v):
    return f"{v:.2f}".replace("-", MINUS)


def build(tmp_path, *extra):
    out = tmp_path / "index.html"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_report.py"), "--out", str(out), *extra],
        check=True,
        cwd=ROOT,
    )
    return out.read_text()


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    return build(tmp_path_factory.mktemp("report"))


def text_of(page):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", page)))


def test_hero_is_the_robust_comparison_not_a_noisy_level(page):
    # Owner rule: the hero states a positive, conditioned result. The level of
    # either Sharpe is within one standard error of zero, so the hero is the
    # count of lag x cost settings the rolling hedge won; if it ever wins none,
    # the hero must change.
    cells = RESULTS["sweep"]
    ahead = sum(c["Rolling OLS"]["test_sharpe"] > c["Kalman"]["test_sharpe"] for c in cells)
    assert ahead > 0
    hero = re.search(r'<p class="hero-metric"><span class="value">([^<]+)</span>', page).group(1)
    assert hero == f"{ahead} of {len(cells)}"
    lag0 = RESULTS["lags"]["0"]
    ols, kal = lag0["Rolling OLS"]["test"]["sharpe"], lag0["Kalman"]["test"]["sharpe"]
    text = text_of(page)
    assert f"Rolling-OLS out-of-sample Sharpe {sh(ols)} at the default setting" in text
    assert f"a gap of {ols - kal:.2f}" in text
    assert f"tested {RESULTS['test_start']} to {RESULTS['test_end']}" in text


def test_numbers_and_units_do_not_wrap_apart(page):
    main = html.unescape(re.sub(r"<[^>]+>", "", page.split("<main", 1)[1]))
    assert not re.search(r"\d[ \n](bp|days?|years?)\b", main)


def test_lag_and_cost_claims_match_the_sweep(page):
    # The same-close fill was the reason to doubt the headline; the page must
    # report the next-day numbers and how many settings keep the ranking.
    text = text_of(page)
    lag1 = RESULTS["lags"]["1"]
    assert f"{sh(lag1['Rolling OLS']['test']['sharpe'])} with next-day fills" in text
    cells = RESULTS["sweep"]
    ahead = sum(c["Rolling OLS"]["test_sharpe"] > c["Kalman"]["test_sharpe"] for c in cells)
    assert f"{ahead} of {len(cells)}" in text
    lo = min(c["Rolling OLS"]["test_sharpe"] for c in cells)
    assert sh(lo) in text, "the worst rolling-OLS setting must be disclosed"
    assert {c["lag"] for c in cells} == {0, 1}


def test_scan_counts(page):
    scan = RESULTS["scan"]
    assert len(scan) == 55  # C(11, 2)
    raw = sum(s["p_value"] <= 0.05 for s in scan)
    bh = sum(s["passes_bh"] for s in scan)
    assert f"turns {raw} raw hits at p ≤ 0.05 into {bh}" in text_of(page)


def test_test_window_length_and_standard_error(page):
    eq = RESULTS["lags"]["0"]["Rolling OLS"]["equity"]
    days = sum(1 for d, _ in eq if d >= RESULTS["test_start"])
    sr = RESULTS["lags"]["0"]["Rolling OLS"]["test"]["sharpe"]
    se = ((1 + sr * sr / 2) / (days / 252)) ** 0.5
    assert f"± {se:.2f} standard error" in text_of(page)


def test_every_evidence_reference_resolves(page):
    refs = set(re.findall(r'href="#evidence-(e\d+)"', page))
    rows = set(re.findall(r'id="evidence-(e\d+)"', page))
    assert refs and refs <= rows
    assert "/blob/main/" not in page


def test_committed_page_and_readme_are_current(page, tmp_path):
    assert page == (ROOT / "docs" / "index.html").read_text(), (
        "docs/index.html is stale: run python3 scripts/build_report.py --readme README.md"
    )
    readme = tmp_path / "README.md"
    readme.write_text((ROOT / "README.md").read_text())
    build(tmp_path, "--readme", str(readme))
    assert readme.read_text() == (ROOT / "README.md").read_text(), "README results block is stale"
