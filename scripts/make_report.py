"""
Write reports/index.html and reports/results.md: one real pair, tuned on a
train window and reported on a test window, plus a synthetic sanity check.

    python scripts/make_report.py --prices data/PEP_KO_2018-01-01_2024-12-31.csv  # CI
    python scripts/make_report.py                    # PEP/KO, 2018-01-01 to 2024-12-31
    python scripts/make_report.py --y XOM --x CVX --start 2015-01-01 --end 2024-12-31
    python scripts/make_report.py --synthetic-only   # offline
    python scripts/make_report.py --card             # also site/card.json, from a clean tree

Without --prices the real pair is downloaded from Yahoo Finance at run time.
Yahoo re-adjusts history after dividends and often blocks CI runners, so the
published report is built from the committed CSV; refresh it with
    python scripts/make_report.py --save-prices data/PEP_KO_2018-01-01_2024-12-31.csv
"""

import argparse
import inspect
import json
import platform
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import charts  # noqa: E402
from src.backtester import Backtester  # noqa: E402
from src.data import close_prices, select_pair  # noqa: E402
from src.metrics import metrics_table  # noqa: E402
from src.research import DEFAULT_TRAIN_FRACTION, evaluate, param_grid, takeaway  # noqa: E402
from src.synthetic import make_pair  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports"
CARD = ROOT / "site" / "card.json"
CARD_MAX_POINTS = 500
TAKEAWAY_MAX = 140
# Synthetic pair: 1,500 bars, beta 1.5, AR(1) spread with phi 0.95 (half-life
# 13.5 bars), noise 1% of the price level. Seeded so the numbers are stable.
PRICES_REL = "data/PEP_KO_2018-01-01_2024-12-31.csv"
_DEFAULTS = inspect.signature(Backtester).parameters
COMMISSION_BPS = _DEFAULTS["commission_bps"].default
SLIPPAGE_BPS = _DEFAULTS["slippage_bps"].default
SYNTHETIC = dict(n=1500, beta=1.5, alpha=0.2, noise=0.01, phi=0.95, level=50.0, seed=42)


def results_table(ev):
    return metrics_table(
        {
            f"{name}, {w}": getattr(run, w)
            for name, run in ev.runs.items()
            for w in ("train", "test")
        }
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


def downsample(series, limit=CARD_MAX_POINTS):
    step = max(1, -(-len(series) // limit))
    return series.iloc[::step]


def build_card(ev, ticker_y, ticker_x, sha, command, generated_at):
    """Portfolio-site card (schema_version 1): test-window net equity for both models."""
    k_sharpe = ev.runs["Kalman"].test["sharpe"]
    o_sharpe = ev.runs["Rolling OLS"].test["sharpe"]
    verb = "beat" if k_sharpe > o_sharpe else "trailed"
    card_takeaway = (
        f"Out of sample, the Kalman hedge {verb} rolling OLS by {abs(k_sharpe - o_sharpe):.2f} "
        f"Sharpe ({k_sharpe:.2f} vs {o_sharpe:.2f}) after costs."
    )
    if len(card_takeaway) > TAKEAWAY_MAX:
        raise ValueError(f"takeaway has {len(card_takeaway)} characters, limit {TAKEAWAY_MAX}")
    series = []
    for name, style in (("Kalman", "solid"), ("Rolling OLS", "dashed")):
        equity = ev.runs[name].result["equity"].loc[ev.test_start : ev.test_end]
        points = [[f"{d:%Y-%m-%d}", round(float(v), 2)] for d, v in downsample(equity).items()]
        series.append({"name": name, "style": style, "points": points})
    return {
        "schema_version": 1,
        "repo": "statarb",
        "sha": sha,
        "generated_at": generated_at,
        "command": command,
        "data_note": (
            f"{ticker_y}/{ticker_x}, test window {ev.test_start:%Y-%m-%d} to "
            f"{ev.test_end:%Y-%m-%d}. Yahoo Finance daily closes; equity only, no prices."
        ),
        "metric": {
            "label": "Out-of-sample Sharpe after costs",
            "value": float(k_sharpe),
            "unit": "",
            "higher_is_better": True,
        },
        "takeaway": card_takeaway,
        "chart": {
            "type": "line",
            "x_label": "Date",
            "x_unit": "",
            "x_scale": "time",
            "y_label": "Net equity",
            "y_unit": "USD",
            "y_scale": "linear",
            "series": series,
        },
    }


def git(*args):
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True, cwd=ROOT
    ).stdout.strip()


def download_closes(ticker_y, ticker_x, start, end):
    import yfinance as yf

    raw = yf.download([ticker_x, ticker_y], start=start, end=end, auto_adjust=True, progress=False)
    return close_prices(raw)[[ticker_y, ticker_x]]


def save_prices(closes, path, ticker_y, ticker_x, start, end):
    """Write the adjusted closes and a provenance file next to them. Yahoo
    re-adjusts history after each dividend, so a cached copy is what makes the
    report reproducible."""
    import yfinance as yf

    path.parent.mkdir(parents=True, exist_ok=True)
    closes.rename_axis("date").to_csv(path)
    provenance = {
        "source": "Yahoo Finance daily closes, adjusted for splits and dividends",
        "tool": f"yfinance {yf.__version__}, yf.download(auto_adjust=True)",
        "tickers": [ticker_y, ticker_x],
        "start": start,
        "end": end,
        "downloaded": datetime.now(UTC).strftime("%Y-%m-%d"),
        "command": "python scripts/make_report.py --save-prices " + path.as_posix(),
    }
    path.with_suffix(".json").write_text(json.dumps(provenance, indent=1) + "\n")


def load_prices(path):
    closes = pd.read_csv(path, index_col="date", parse_dates=["date"])
    provenance = json.loads(path.with_suffix(".json").read_text())
    return closes, provenance


def real_pair(closes, ticker_y, ticker_x):
    return select_pair(closes, ticker_y=ticker_y, ticker_x=ticker_x).dropna()


REPO_URL = "https://github.com/allenwang0/statarb"
DOCS = ROOT / "docs"
KIT = DOCS / "kit"
BETA_EVERY = 5  # beta chart: every 5th bar (about weekly) keeps the page small


def _signed(v, fmt="{:.2f}"):
    return fmt.format(v).replace("-", "−")


def _pct(v):
    return _signed(v, "{:.1f}") + " %"


def _iso(d):
    return f"{d:%Y-%m-%d}"


def _test_curve(run, ev):
    """Test-window cumulative return, from the equity on the bar before the
    test starts (the same base the test metrics use)."""
    equity = run.result["equity"]
    base = equity.loc[: ev.train_end].iloc[-1]
    return equity.loc[ev.test_start : ev.test_end] / base - 1


def _metrics_rows(ev):
    table = results_table(ev)
    return {"columns": list(table.columns), "rows": [[k, *v] for k, v in table.iterrows()]}


def page_data(ev, ticker_y, ticker_x):
    """Everything the charts draw, computed from the evaluation."""
    kalman, ols = ev.runs["Kalman"], ev.runs["Rolling OLS"]
    curves = {name: _test_curve(run, ev) for name, run in ev.runs.items()}
    dates = curves["Kalman"].index
    k_res, o_res = kalman.result, ols.result
    beta = k_res.loc[ev.train_start :].iloc[::BETA_EVERY]
    o_beta = o_res.loc[beta.index]
    z = k_res["z_score"].dropna().to_numpy()
    acf = charts.autocorrelation(z)
    return {
        "pair": [ticker_y, ticker_x],
        "test": {
            "t": [int(d.timestamp()) for d in dates],
            "kalman": [round(float(v) * 100, 3) for v in curves["Kalman"]],
            "ols": [round(float(v) * 100, 3) for v in curves["Rolling OLS"]],
        },
        "beta": {
            "t": [int(d.timestamp()) for d in beta.index],
            "kalman": [round(float(v), 4) for v in beta["hedge_ratio"]],
            "lo": [round(float(v), 4) for v in beta["hedge_ratio"] - 2 * beta["beta_sd"]],
            "hi": [round(float(v), 4) for v in beta["hedge_ratio"] + 2 * beta["beta_sd"]],
            "ols": [round(float(v), 4) for v in o_beta["hedge_ratio"]],
        },
        "acf": {
            "lag": list(range(1, len(acf) + 1)),
            "v": [round(float(v), 4) for v in acf],
            "band": round(float(charts.Z_95 / np.sqrt(len(z))), 4),
        },
        "split": [int(ev.test_start.timestamp())],
    }


def diagnostics(ev):
    z = ev.runs["Kalman"].result["z_score"].dropna().to_numpy()
    acf = charts.autocorrelation(z)
    return {
        "variance": float(np.var(z)),
        "outside": int((np.abs(acf) > charts.Z_95 / np.sqrt(len(z))).sum()),
        "lags": charts.ACF_LAGS,
        "n": len(z),
    }


def _card(span, label, metric, cmp, foot):
    return (
        f'<div class="card span-{span}"><span class="label">{escape(label)}</span>'
        f'<span class="metric num">{escape(metric)}</span>'
        f'<span class="cmp">{escape(cmp)}</span><span class="foot">{escape(foot)}</span></div>'
    )


def _figure(fid, title, sub, legend="", caption=""):
    return (
        f'<figure class="figure" aria-labelledby="f-{fid}"><div class="fig-head"><div>'
        f'<h3 id="f-{fid}">{escape(title)}</h3><p class="sub">{escape(sub)}</p></div>'
        '<div class="seg" role="radiogroup" aria-label="View" data-view-toggle>'
        '<button role="radio" aria-checked="true" data-value="chart">Chart</button>'
        '<button role="radio" aria-checked="false" data-value="table">Table</button></div></div>'
        f'{legend}<div id="c-{fid}"></div><div id="t-{fid}" hidden></div>'
        + (f"<figcaption>{caption}</figcaption>" if caption else "")
        + "</figure>"
    )


def _section(sid, eyebrow, h2, lede, body):
    return (
        f'<section class="s wrap" id="{sid}" aria-labelledby="h-{sid}">'
        f'<p class="eyebrow">{escape(eyebrow)}</p><h2 id="h-{sid}">{escape(h2)}</h2>'
        f'<p class="lede">{escape(lede)}</p>{body}</section>'
    )


def _caveat(lead, text):
    return f'<p class="caveat" role="note"><b>{escape(lead)}</b> {escape(text)}</p>'


def _table(caption, columns, rows, cls=""):
    head = "".join(f'<th scope="col">{escape(c)}</th>' for c in columns)
    body = "".join(
        f'<tr><th scope="row">{escape(r[0])}</th>'
        + "".join(f"<td>{escape(str(v))}</td>" for v in r[1:])
        + "</tr>"
        for r in rows
    )
    return (
        f'<div class="tablewrap {cls}" tabindex="0" role="region" aria-label="{escape(caption)}">'
        f"<table><caption>{escape(caption)}</caption><thead><tr>{head}</tr></thead>"
        f"<tbody>{body}</tbody></table></div>"
    )


def render_page(real, synth, sha, date, prices_note):
    """real: (ev, ticker_y, ticker_x, provenance) or None; synth: evaluation."""
    parts, data = [], {}
    cost_text = f"{COMMISSION_BPS:g} bp commission + {SLIPPAGE_BPS:g} bp slippage per fill"

    if real:
        ev, ty, tx, prov = real
        k, o = ev.runs["Kalman"].test, ev.runs["Rolling OLS"].test
        s, dg = ev.stats, diagnostics(ev)
        lost = k["sharpe"] < o["sharpe"]
        pair = f"{ty}/{tx}"
        h1 = (
            f"On {pair}, the Kalman hedge lost to a plain rolling hedge out of sample"
            if lost
            else f"On {pair}, the Kalman hedge beat a plain rolling hedge out of sample"
        )
        fails = not s.cointegrated
        lede = (
            f"Test-window Sharpe {_signed(k['sharpe'])} for Kalman vs {_signed(o['sharpe'])} "
            f"for rolling OLS, after costs. "
            + (
                f"The pair also fails the Engle-Granger cointegration test on the training "
                f"window (p = {s.eg_pvalue:.3f}), so neither result shows a tradeable edge."
                if fails
                else f"The pair passes the Engle-Granger test on the training window "
                f"(p = {s.eg_pvalue:.3f})."
            )
        )
        conds = [
            f'<span class="key">Real prices: {escape(prov["source"].split(",")[0])}, '
            f"downloaded {escape(prov['downloaded'])}</span>",
            f"<span>Test {_iso(ev.test_start)} to {_iso(ev.test_end)}</span>",
            f"<span>One pair, one {DEFAULT_TRAIN_FRACTION:.0%} / "
            f"{1 - DEFAULT_TRAIN_FRACTION:.0%} split, no confidence interval</span>",
            f"<span>{escape(cost_text)}</span>",
        ]
        data["real"] = page_data(ev, ty, tx)
        parts.append(
            '<section class="hero wrap" aria-labelledby="h1">'
            '<p class="eyebrow">Comparative study · Python</p>'
            f'<h1 id="h1">{escape(h1)}</h1><p class="lede">{escape(lede)}</p>'
            f'<p class="conditions">{"".join(conds)}</p>'
            + _figure(
                "test",
                "Rolling OLS ends the test window ahead"
                if lost
                else "Kalman ends the test window ahead",
                f"Cumulative return on the test window, % · {pair} · after costs",
                caption="Return since the last bar of the training window, net of costs. "
                "Both hedges use parameters picked on the training window only. One window, "
                "so there is no spread across runs.",
            )
            + "</section>"
        )
        cards = [
            _card(
                4,
                "Test Sharpe, Kalman hedge",
                _signed(k["sharpe"]),
                f"vs {_signed(o['sharpe'])} for rolling OLS",
                f"{k['trades']} vs {o['trades']} trades · after costs",
            ),
            _card(
                4,
                "Cointegration, training window",
                f"p = {s.eg_pvalue:.3f}",
                "Engle-Granger; " + ("fails at 0.05" if fails else "passes at 0.05"),
                f"ADF p (spread) {s.adf_pvalue:.3f} · half-life {s.half_life:.1f} days",
            ),
            _card(
                4,
                "Kalman innovation variance",
                f"{dg['variance']:.2f}",
                "≈ 1.00 if the model fit",
                f"{dg['outside']} of {dg['lags']} autocorrelations outside the 95 % band",
            ),
        ]
        parts.append(
            '<section class="wrap" aria-label="Key results"><div class="bento">'
            + "".join(cards)
            + "</div></section>"
        )

        # Out-of-sample comparison
        m = _metrics_rows(ev)
        h2 = (
            f"Rolling OLS returned {_pct(o['annual_return'] * 100)} a year on the test window; "
            f"Kalman returned {_pct(k['annual_return'] * 100)}"
        )
        tm = metrics_table({"Kalman": k, "Rolling OLS": o})
        body = _table(
            f"Test window {_iso(ev.test_start)} to {_iso(ev.test_end)}, after costs",
            ["Metric", "Kalman", "Rolling OLS"],
            [[name, *row] for name, row in tm.iterrows()],
        )
        body += (
            '<p class="note">Sharpe and Sortino use daily returns, annualized with 252 days and '
            "a zero risk-free rate. Hit rate counts closed trades only.</p>"
        )
        body += _caveat(
            "One window.",
            f"Kalman traded {k['trades']} times and rolling OLS {o['trades']} times in "
            "one test window, with no bootstrap, so the Sharpe gap has no confidence interval.",
        )
        body += (
            f"<details><summary>Train and test metrics, both hedges · {len(m['rows'])} rows"
            "</summary>"
            + _table("Train and test metrics", ["Metric", *m["columns"]], m["rows"])
            + f"<p class='note'>{escape(params_line(ev))}</p></details>"
        )
        data["real"]["metrics"] = m
        parts.append(
            _section(
                "oos",
                "Which hedge did better out of sample?",
                h2,
                f"Both hedges run through the same signal, sizing and cost code. Kalman's "
                f"max drawdown was {_pct(k['max_drawdown'] * 100)} against "
                f"{_pct(o['max_drawdown'] * 100)} for rolling OLS.",
                body,
            )
        )

        # Why: hedge ratio
        kb = ev.runs["Kalman"].result["hedge_ratio"].loc[ev.train_start :]
        ob = ev.runs["Rolling OLS"].result["hedge_ratio"].loc[ev.train_start :]
        jump = ob.diff().abs().mean() / kb.diff().abs().mean()
        window = ev.runs["Rolling OLS"].params.get("ols_window")
        band = float((4 * ev.runs["Kalman"].result["beta_sd"].loc[ev.train_start :]).mean())
        parts.append(
            _section(
                "beta",
                "How do the two hedges differ?",
                f"Rolling OLS's hedge ratio moves {jump:.1f}× as much per day as Kalman's",
                f"The {window}-day OLS window re-fits the ratio from scratch every day; the "
                "Kalman filter adjusts it a little at a time and carries an uncertainty band.",
                _figure(
                    "beta",
                    "Hedge ratio over the full period",
                    f"Shares of {tx} per share of {ty} · every {BETA_EVERY}th trading day",
                    caption="The Kalman estimate carries a ± 2 standard deviation band, "
                    f"{band:.2f} wide on average, so it is barely visible at this scale. Shown from "
                    f"the end of the warm-up ({_iso(ev.train_start)}); the test "
                    f"window starts {_iso(ev.test_start)}.",
                ),
            )
        )

        # Why: diagnostics
        parts.append(
            _section(
                "fit",
                "Does the Kalman model fit this pair?",
                f"No: its standardized errors have variance {dg['variance']:.2f}, not ≈ 1, "
                f"and stay autocorrelated at {dg['outside']} of {dg['lags']} lags",
                "If the model fit, each day's prediction error divided by its predicted "
                "standard deviation would look like independent draws with variance 1. "
                "Errors this large and persistent mean the z-score signal is miscalibrated.",
                _figure(
                    "acf",
                    "Autocorrelation of the standardized errors",
                    f"Lags 1–{dg['lags']} days · n = {dg['n']:,} days · full period",
                    caption="The horizontal lines mark the 95 % band for white noise "
                    "(± 1.96 / √n). Points outside it are autocorrelation the model leaves "
                    "unexplained.",
                ),
            )
        )

    # Sanity check appendix (synthetic)
    sk, so = synth.runs["Kalman"].test, synth.runs["Rolling OLS"].test
    sm = _metrics_rows(synth)
    hit_note = (
        f"Rolling OLS's {so['hit_rate'] * 100:.0f} % hit rate comes from only "
        f"{so['trades']} trades, so it says little about the method."
        if so["trades"] < 30
        else ""
    )
    parts.append(
        _section(
            "sanity",
            "Appendix · Sanity check",
            "On a constructed cointegrated pair, both hedges profit, as they should",
            f"Synthetic data (generated in code, seed {SYNTHETIC['seed']}), not market prices. "
            f"Test Sharpe {_signed(sk['sharpe'])} for Kalman and {_signed(so['sharpe'])} for "
            "rolling OLS. This checks the code, not an edge: the spread mean-reverts by "
            "construction.",
            (_caveat("Few trades.", hit_note) if hit_note else "")
            + f"<details><summary>Synthetic pair metrics · {len(sm['rows'])} rows</summary>"
            + _table(
                "Synthetic pair, train and test metrics", ["Metric", *sm["columns"]], sm["rows"]
            )  # fmt: skip
            + f"<p class='note'>{escape(stats_line(synth))}</p>"
            f"<p class='note'>{escape(params_line(synth))}</p></details>",
        )
    )

    # Methodology
    n_k, n_o = len(param_grid("kalman")), len(param_grid("ols"))
    parts.append(
        '<section class="s wrap" id="method" aria-labelledby="h-method">'
        '<p class="eyebrow">Methodology</p><h2 id="h-method">How this was tested</h2>'
        '<div class="prose" style="margin-top:24px">'
        f"<p>The first {DEFAULT_TRAIN_FRACTION:.0%} of the bars are the training window and "
        "the rest the test window. Each hedge's entry threshold, exit threshold and speed "
        f"(Kalman delta or OLS window) are picked by Sharpe on the training window from a "
        f"grid of {n_k} (Kalman) or {n_o} (OLS) settings. Nothing after the training window "
        "touches that choice.</p>"
        "<p>Both hedges then trade the whole series with those settings. Positions open "
        "when the z-score passes the entry threshold and close when it returns past the exit "
        f"threshold, sized at account equity. Costs: {escape(cost_text)}.</p></div>"
        '<dl class="kv prose"><dt>Hedge ratio</dt><dd>Shares of X held against one share of '
        "Y.</dd><dt>Engle-Granger p</dt><dd>Chance of a spread this stationary if the pair "
        "were not cointegrated; above 0.05 fails.</dd><dt>Innovation</dt><dd>The filter's "
        "one-day prediction error; divided by its predicted standard deviation it should "
        "have variance 1.</dd><dt>Sharpe</dt><dd>Annualized mean over standard deviation of "
        "daily returns, risk-free rate 0.</dd></dl>"
        '<div class="two"><div><h3>What this shows</h3><ul>'
        "<li>On this pair and window, the Kalman hedge did worse than a plain rolling hedge "
        "after costs.</li><li>The diagnostics explain why: the pair isn't cointegrated in "
        "training and the filter's error model doesn't fit.</li>"
        "<li>The code behaves as designed on a pair built to mean-revert.</li></ul></div>"
        "<div><h3>What it doesn't show</h3><ul>"
        "<li>That either hedge has a tradeable edge.</li>"
        "<li>How the gap varies across pairs or windows: one pair, one split.</li>"
        "<li>Uncertainty in the Sharpe values: no bootstrap or repeated windows.</li>"
        "<li>Borrow fees, margin interest or fills away from the close.</li></ul></div></div>"
        "</section>"
    )

    # Reproduce
    env = [("Prices", prices_note)] if real else []
    if real:
        env.append(("Provenance", real[3]["tool"]))
    env += [
        ("Synthetic pair", ", ".join(f"{k} {v}" for k, v in SYNTHETIC.items())),
        ("Python", platform.python_version()),
        ("numpy / pandas / statsmodels",
         f"{np.__version__} / {pd.__version__} / {statsmodels.__version__}"),
        ("Commit", sha or "unknown"),
    ]  # fmt: skip
    parts.append(
        '<section class="s wrap" id="reproduce" aria-labelledby="h-repro">'
        '<p class="eyebrow">Reproduce</p><h2 id="h-repro">Run it yourself</h2>'
        '<div class="codewrap" style="margin-top:24px"><pre><code>'
        f"git clone {REPO_URL}.git &amp;&amp; cd statarb\n"
        "python3.12 -m venv .venv   # Python 3.11 or 3.12\n"
        '.venv/bin/pip install -e ".[dev]"\n'
        f".venv/bin/python scripts/make_report.py --prices {escape(PRICES_REL)}\n"
        ".venv/bin/streamlit run app.py   # the interactive app"
        "</code></pre></div>"
        + _table("Environment for this build", ["Field", "Value"], env, cls="env")
        + "</section>"
    )

    data["split_label"] = "test starts"
    footer = f'Data from commit <span class="mono">{escape(sha or "unknown")}</span>' + (
        f" · {escape(date)}" if date else ""
    )

    def kit(name):
        return (KIT / name).read_text()

    page = (DOCS / "report.template.html").read_text()
    for key, value in (
        ("%%KIT_CSS%%", kit("report.css")),
        ("%%LOCAL_CSS%%", (DOCS / "report.local.css").read_text()),
        ("%%KIT_JS%%", kit("report.js") + "\n" + kit("charts.js")),
        ("%%LOCAL_JS%%", (DOCS / "report.local.js").read_text()),
        ("%%REPO%%", REPO_URL),
        ("%%FOOTER%%", footer),
        ("%%BODY%%", "\n".join(parts)),
        ("%%DATA%%", json.dumps(data, separators=(",", ":")).replace("</", "<\\/")),
    ):
        page = page.replace(key, value)
    return page


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--y", default="PEP")
    parser.add_argument("--x", default="KO")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--prices", type=Path, help="cached closes CSV (skips the download)")
    parser.add_argument(
        "--save-prices", type=Path, help="download, save the closes CSV + provenance, and use it"
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--card", action="store_true", help="also write site/card.json")
    args = parser.parse_args()
    if args.card and (args.synthetic_only or git("status", "--porcelain")):
        sys.exit("--card needs the real pair and a clean git tree, so its sha matches the code")

    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    date = subprocess.run(
        ["git", "log", "-1", "--format=%cs"], capture_output=True, text=True, check=False
    ).stdout.strip()
    header = f"Generated from commit {sha or 'unknown'} by scripts/make_report.py."

    md = [header]
    synth = evaluate(make_pair(**SYNTHETIC))
    label = f"Synthetic data (generated in code, seed {SYNTHETIC['seed']}), not market prices."
    md.append(markdown("Synthetic pair (sanity check)", label, synth))

    real = None
    if not args.synthetic_only:
        if args.prices:
            closes, prov = load_prices(args.prices)
            prices_note = f"{args.prices.as_posix()}, downloaded {prov['downloaded']}"
            args.y, args.x = prov["tickers"]
            args.start, args.end = prov["start"], prov["end"]
        else:
            closes = download_closes(args.y, args.x, args.start, args.end)
            if args.save_prices:
                save_prices(closes, args.save_prices, args.y, args.x, args.start, args.end)
                closes, prov = load_prices(args.save_prices)
            else:
                prov = {
                    "source": "Yahoo Finance daily closes, adjusted for splits and dividends",
                    "tool": "yfinance, yf.download(auto_adjust=True)",
                    "downloaded": datetime.now(UTC).strftime("%Y-%m-%d"),
                }
            prices_note = f"downloaded {prov['downloaded']} at report time"
        data = real_pair(closes, args.y, args.x)
        ev = evaluate(data)
        label = (
            f"{args.y} (Y) against {args.x} (X), {prov['source']}, {args.start} to {args.end}, "
            f"downloaded {prov['downloaded']}. Costs: {COMMISSION_BPS:g} bp commission and "
            f"{SLIPPAGE_BPS:g} bp slippage per fill."
        )
        title = f"{args.y} / {args.x}"
        md.insert(1, markdown(title, label, ev))
        real = (ev, args.y, args.x, prov)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    page = render_page(real, synth, sha, date, prices_note if real else "")
    (out_dir / "index.html").write_text(page)
    (out_dir / "results.md").write_text("\n\n".join(md) + "\n")
    print("\n\n".join(md))
    print(f"\nWrote {out_dir / 'index.html'} and {out_dir / 'results.md'}")

    if args.card:
        command = " ".join(
            shlex.quote(a) for a in ["python", "scripts/make_report.py", *sys.argv[1:]]
        )
        card = build_card(
            ev,
            args.y,
            args.x,
            sha=git("rev-parse", "HEAD"),
            command=command,
            generated_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        CARD.parent.mkdir(exist_ok=True)
        CARD.write_text(json.dumps(card, indent=1) + "\n")
        print(f"Wrote {CARD.relative_to(ROOT)}: {card['takeaway']}")


if __name__ == "__main__":
    main()
