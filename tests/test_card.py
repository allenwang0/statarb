import importlib.util
import json
from pathlib import Path

import jsonschema
import pytest

from src.research import evaluate
from src.synthetic import make_pair

ROOT = Path(__file__).resolve().parents[1]
# Written from the schema_version 1 contract in allenwang0.github.io's plan;
# that repo does not have scripts/card.schema.json yet.
SCHEMA = json.loads((ROOT / "tests" / "fixtures" / "card.schema.json").read_text())
_spec = importlib.util.spec_from_file_location("make_report", ROOT / "scripts" / "make_report.py")
make_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_report)


def check_card(card):
    jsonschema.validate(card, SCHEMA)
    kalman, ols = card["chart"]["series"]
    assert [p[0] for p in kalman["points"]] == [p[0] for p in ols["points"]]
    assert (kalman["name"], kalman["style"], ols["name"], ols["style"]) == (
        "Kalman", "solid", "Rolling OLS", "dashed",
    )  # fmt: skip


def test_card_metric_is_the_reports_kalman_sharpe():
    # The site shows this number as the headline, so it must be the
    # out-of-sample Kalman Sharpe the report prints, not a recomputation.
    ev = evaluate(make_pair(**make_report.SYNTHETIC))
    card = make_report.build_card(
        ev, "SYN_Y", "SYN_X", "0" * 40, "python scripts/make_report.py --card",
        "2026-01-01T00:00:00Z",
    )  # fmt: skip
    check_card(card)
    assert card["metric"]["value"] == ev.runs["Kalman"].test["sharpe"]
    assert len(card["chart"]["series"][0]["points"]) <= make_report.CARD_MAX_POINTS
    assert card["chart"]["series"][0]["points"][0][0] == f"{ev.test_start:%Y-%m-%d}"


def test_committed_card_validates():
    path = ROOT / "site" / "card.json"
    if not path.exists():
        pytest.fail("site/card.json is missing; run python scripts/make_report.py --card")
    check_card(json.loads(path.read_text()))
