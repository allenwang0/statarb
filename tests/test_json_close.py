import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("json_close", ROOT / "scripts" / "json_close.py")
json_close = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(json_close)


def test_floats_compare_within_tolerance_only():
    # CI uses this to accept last-digit float noise from another machine's
    # BLAS, so it must still catch a real change in a result.
    assert json_close.diff({"sharpe": 0.616657}, {"sharpe": 0.616657 + 1e-12}) == []
    assert json_close.diff({"sharpe": 0.616657}, {"sharpe": 0.616658}, atol=1e-9)


def test_everything_else_is_exact():
    assert json_close.diff({"trades": 55}, {"trades": 56})
    assert json_close.diff({"date": "2022-03-14"}, {"date": "2022-03-15"})
    assert json_close.diff({"passes_bh": False}, {"passes_bh": True})
    assert json_close.diff([1.0, 2.0], [1.0])
    assert json_close.diff({"a": 1.0}, {"b": 1.0})
