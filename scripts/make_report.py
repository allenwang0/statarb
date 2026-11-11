"""
Write reports/report.html and reports/results.md: a synthetic pair and one
real pair, each tuned on a train window and reported on a test window.

    python scripts/make_report.py                    # PEP/KO, 2018-01-01 to 2024-12-31
    python scripts/make_report.py --y XOM --x CVX --start 2015-01-01 --end 2024-12-31
    python scripts/make_report.py --synthetic-only   # offline

The real-pair section downloads from Yahoo Finance at run time; reports/ is
git-ignored so no prices are committed.
"""

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import plotly.io as pio
from plotly.offline import get_plotlyjs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import charts  # noqa: E402
from src.data import close_prices, select_pair  # noqa: E402
from src.metrics import metrics_table  # noqa: E402
from src.research import evaluate, takeaway  # noqa: E402
from src.synthetic import make_pair  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "reports"
# Synthetic pair: 1,500 bars, beta 1.5, AR(1) spread with phi 0.95 (half-life
# 13.5 bars), noise 1% of the price level. Seeded so the numbers are stable.
SYNTHETIC = dict(n=1500, beta=1.5, alpha=0.2, noise=0.01, phi=0.95, level=50.0, seed=42)


def section(title, label, data, ev, ticker_y, ticker_x):
    figs = [
        charts.equity_and_drawdown(ev.runs, ev),
        charts.prices_and_spread(data, ev.runs["Kalman"].result, ev, ticker_y, ticker_x),
        charts.zscore_chart(
            ev.runs["Kalman"].result,
            ev.runs["Kalman"].params["entry_threshold"],
            ev.runs["Kalman"].params["exit_threshold"],
            "Kalman",
        ),
        charts.hedge_ratio_chart(
            ev.runs["Kalman"].result, ev.runs["Rolling OLS"].result, ticker_y, ticker_x
        ),
        charts.innovation_diagnostics(ev.runs["Kalman"].result),
    ]
    html = [f"<h2>{title}</h2>", f"<p><strong>{label}</strong></p>", f"<p>{takeaway(ev)}</p>"]
    html.append(f"<p>{stats_line(ev)}</p>")
    html.append(results_table(ev).to_html())
    html.append(f"<p>{params_line(ev)}</p>")
    html.append(f"<p>{charts.diagnostics_takeaway(ev.runs['Kalman'].result)}</p>")
    for fig in figs:
        fig.update_layout(template="plotly_white")
        html.append(pio.to_html(fig, include_plotlyjs=False, full_html=False))
    return "\n".join(html)


def results_table(ev):
    return metrics_table(
        {f"{name}, {w}": getattr(run, w) for name, run in ev.runs.items() for w in ("train", "test")}
    )


def stats_line(ev):
    s = ev.stats
    return (
        f"Train window {ev.train_start:%Y-%m-%d} to {ev.train_end:%Y-%m-%d}, test window "
        f"{ev.test_start:%Y-%m-%d} to {ev.test_end:%Y-%m-%d}. On the train window: "
        f"Engle-Granger p = {s.eg_pvalue:.3f}, ADF p (spread) = {s.adf_pvalue:.3f}, "
        f"half-life = {s.half_life:.1f} days."
    )


def params_line(ev):
    return "Parameters chosen on the train window: " + "; ".join(
        f"{name}: " + ", ".join(f"{k}={v:g}" for k, v in run.params.items())
        for name, run in ev.runs.items()
    )


def markdown(title, label, ev):
    return "\n\n".join(
        [f"## {title}", label, stats_line(ev), to_markdown(results_table(ev)), params_line(ev),
         takeaway(ev)]
    )  # fmt: skip


def to_markdown(table):
    header = "| Metric | " + " | ".join(table.columns) + " |"
    rule = "|---" * (len(table.columns) + 1) + "|"
    rows = [f"| {name} | " + " | ".join(row) + " |" for name, row in table.iterrows()]
    return "\n".join([header, rule, *rows])


def real_pair(ticker_y, ticker_x, start, end):
    import yfinance as yf

    raw = yf.download(
        [ticker_x, ticker_y], start=start, end=end, auto_adjust=True, progress=False
    )
    return select_pair(close_prices(raw), ticker_y=ticker_y, ticker_x=ticker_x).dropna()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--y", default="PEP")
    parser.add_argument("--x", default="KO")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    run_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    header = f"Generated {run_at} from commit {sha or 'unknown'} by scripts/make_report.py."

    sections, md = [], [header]
    data = make_pair(**SYNTHETIC)
    ev = evaluate(data)
    label = "Synthetic data (generated in code, seed 42), not market prices."
    sections.append(section("Synthetic pair", label, data, ev, "SYN_Y", "SYN_X"))
    md.append(markdown("Synthetic pair", label, ev))

    if not args.synthetic_only:
        data = real_pair(args.y, args.x, args.start, args.end)
        ev = evaluate(data)
        label = (
            f"{args.y} (Y) against {args.x} (X), Yahoo Finance adjusted daily closes, "
            f"{args.start} to {args.end}, downloaded at report time. "
            f"Costs: 1 bp commission and 1 bp slippage per fill."
        )
        title = f"{args.y} / {args.x}"
        sections.append(section(title, label, data, ev, args.y, args.x))
        md.append(markdown(title, label, ev))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    html = (
        "<!doctype html><html><head><meta charset='utf-8'><title>statarb report</title>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<script>{get_plotlyjs()}</script>"
        "<style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:auto;padding:16px}"
        "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 8px}</style>"
        f"</head><body><h1>statarb report</h1><p>{header}</p>{''.join(sections)}</body></html>"
    )
    (out_dir / "report.html").write_text(html)
    (out_dir / "results.md").write_text("\n\n".join(md) + "\n")
    print("\n\n".join(md))
    print(f"\nWrote {out_dir / 'report.html'} and {out_dir / 'results.md'}")


if __name__ == "__main__":
    main()
