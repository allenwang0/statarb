"""
Write results/case_study.json: the PEP/KO train/test evaluation at fill lags
0 and 1, a cost sweep, the hedge-ratio paths, and the 55-pair scan, all from
the committed price snapshot data/prices.csv so the report rebuilds offline.

    python scripts/case_study.py            # from data/prices.csv
    python scripts/case_study.py --fetch    # re-download the snapshot first (network)

Uses the same code as the app: src.research.evaluate and src.scanner.scan.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import close_prices, select_pair  # noqa: E402
from src.research import DEFAULT_TRAIN_FRACTION, ENTRY_GRID, EXIT_GRID, evaluate  # noqa: E402
from src.scanner import DEFAULT_UNIVERSE, scan  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PRICES = ROOT / "data" / "prices.csv"
META = ROOT / "data" / "prices.meta.json"
OUT = ROOT / "results" / "case_study.json"
START, END = "2018-01-01", "2024-12-31"  # yfinance end is exclusive
PAIR = ("PEP", "KO")  # (Y, X)
LAGS = (0, 1)
COSTS_BPS = (0, 2, 5, 10, 20)  # round-trip per fill: commission + slippage, split evenly
DEFAULT_COST_BPS = 2
MODELS = ("Kalman", "Rolling OLS")


def fetch():
    import yfinance as yf

    tickers = list(DEFAULT_UNIVERSE)
    raw = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)
    prices = close_prices(raw)[list(DEFAULT_UNIVERSE)]
    PRICES.parent.mkdir(exist_ok=True)
    prices.to_csv(PRICES, float_format="%.4f", index_label="Date")
    META.write_text(json.dumps({
        "source": "Yahoo Finance via yfinance, auto_adjust=True (adjusted closes)",
        "yfinance": yf.__version__,
        "tickers": list(DEFAULT_UNIVERSE),
        "start": START,
        "end_exclusive": END,
        "fetched_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rounding": "4 decimals",
    }, indent=1) + "\n")  # fmt: skip


def clean(value):
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else round(float(value), 6)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    return value


def series(s, digits):
    return [[f"{d:%Y-%m-%d}", round(float(v), digits)] for d, v in s.dropna().items()]


def run(data, lag, cost_bps):
    half = cost_bps / 2
    return evaluate(data, fill_lag=lag, commission_bps=half, slippage_bps=half)


def model_summary(ev, name, equity=False):
    r = ev.runs[name]
    out = {"params": clean(r.params), "train": clean(r.train), "test": clean(r.test)}
    if equity:
        out["equity"] = series(r.result["equity"].loc[ev.train_start :], 2)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fetch", action="store_true", help="re-download data/prices.csv")
    args = parser.parse_args()
    if args.fetch:
        fetch()
    prices = pd.read_csv(PRICES, index_col="Date", parse_dates=True)
    data = select_pair(prices, ticker_y=PAIR[0], ticker_x=PAIR[1])

    by_lag = {}
    for lag in LAGS:
        ev = run(data, lag, DEFAULT_COST_BPS)
        by_lag[str(lag)] = {m: model_summary(ev, m, equity=True) for m in MODELS}
        if lag == 0:
            base = ev
    sweep = []
    for lag in LAGS:
        for cost in COSTS_BPS:
            ev = run(data, lag, cost)
            sweep.append({
                "lag": lag,
                "cost_bps": cost,
                **{m: {"test_sharpe": clean(ev.runs[m].test["sharpe"]),
                       "train_sharpe": clean(ev.runs[m].train["sharpe"])} for m in MODELS},
            })  # fmt: skip
    hedge = {m: series(base.runs[m].result["hedge_ratio"], 4) for m in MODELS}
    table = scan(prices[list(DEFAULT_UNIVERSE)])
    s = base.stats
    result = {
        "pair": {"y": PAIR[0], "x": PAIR[1]},
        "data": json.loads(META.read_text()),
        "rows": len(data),
        "first_date": f"{data.index[0]:%Y-%m-%d}",
        "last_date": f"{data.index[-1]:%Y-%m-%d}",
        "train_fraction": DEFAULT_TRAIN_FRACTION,
        "warmup_start": f"{base.warmup_start:%Y-%m-%d}",
        "train_start": f"{base.train_start:%Y-%m-%d}",
        "train_end": f"{base.train_end:%Y-%m-%d}",
        "test_start": f"{base.test_start:%Y-%m-%d}",
        "test_end": f"{base.test_end:%Y-%m-%d}",
        "default_cost_bps": DEFAULT_COST_BPS,
        "grid_points": {"entry": list(ENTRY_GRID), "exit": list(EXIT_GRID)},
        "train_stats": clean(
            {
                "eg_pvalue": s.eg_pvalue,
                "adf_pvalue": s.adf_pvalue,
                "half_life": s.half_life,
                "hedge_ratio": s.hedge_ratio,
            }
        ),
        "lags": by_lag,
        "sweep": sweep,
        "hedge_ratio": hedge,
        "scan": [clean(r) for r in table.to_dict("records")],
        "command": "python scripts/case_study.py",
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    for lag in LAGS:
        k, o = (by_lag[str(lag)][m]["test"]["sharpe"] for m in MODELS)
        print(f"lag {lag}: test Sharpe Kalman {k:.2f}, rolling OLS {o:.2f}")
    print(f"scan: {int(table['passes_bh'].sum())} of {len(table)} pairs pass BH at 5%")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
