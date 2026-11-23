import json
import subprocess
import sys
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
# Written from the schema_version 1 contract in allenwang0.github.io's plan;
# that repo does not have scripts/card.schema.json yet.
SCHEMA = json.loads((ROOT / "tests" / "fixtures" / "card.schema.json").read_text())
RESULTS = json.loads((ROOT / "results" / "case_study.json").read_text())


def check_card(card):
    jsonschema.validate(card, SCHEMA)
    ols, kalman = card["chart"]["series"]
    assert [p[0] for p in ols["points"]] == [p[0] for p in kalman["points"]]
    assert (ols["name"], ols["style"], kalman["name"], kalman["style"]) == (
        "Rolling OLS", "solid", "Kalman", "dashed",
    )  # fmt: skip


def build_card(tmp_path):
    out = tmp_path / "card.json"
    script = str(ROOT / "scripts" / "build_report.py")
    subprocess.run(
        [sys.executable, script, "--out", str(tmp_path / "i.html"), "--card", str(out)],
        check=True,
        cwd=ROOT,
    )
    return json.loads(out.read_text())


def test_card_headlines_the_reports_positive_comparison(tmp_path):
    # The portfolio shows this metric as the project's headline, so it must be
    # the report's hero (settings won out of all tested), recomputed from the
    # committed results, and never a negative Sharpe.
    card = build_card(tmp_path)
    check_card(card)
    cells = RESULTS["sweep"]
    ahead = sum(c["Rolling OLS"]["test_sharpe"] > c["Kalman"]["test_sharpe"] for c in cells)
    assert card["metric"]["value"] == ahead > 0
    assert card["metric"]["higher_is_better"]
    assert f"{ahead} of {len(cells)}" in card["takeaway"]
    assert "−" not in card["takeaway"] and "-0." not in card["takeaway"]
    ols = card["chart"]["series"][0]["points"]
    assert ols[0][0] == RESULTS["test_start"]
    assert len(ols) <= 500


def test_committed_card_is_current(tmp_path):
    assert build_card(tmp_path) == json.loads((ROOT / "site" / "card.json").read_text()), (
        "site/card.json is stale: run python3 scripts/build_report.py --card site/card.json"
    )
