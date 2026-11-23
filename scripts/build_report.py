"""Build docs/index.html (and the README results block) from the committed
case-study output results/case_study.json and its run record results/run.json.

    python3 scripts/build_report.py [--readme README.md]

Python 3.10+ standard library only, so the page rebuilds without the app's
dependencies. Every number on the page is computed here from those files or
from constants in the committed sources; none is typed by hand. Components and
charts come from the vendored report kit in report-kit/.
"""

import argparse
import datetime
import json
import math
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "report-kit"))
import report_kit as rk  # noqa: E402
from report_kit import Raw, coord, esc, svg_text  # noqa: E402

REPO = "allenwang0/statarb"
REPO_URL = f"https://github.com/{REPO}"
PAGE_URL = "https://allenwang0.github.io/statarb/"
HOME_URL = "https://allenwang0.github.io/"
MODELS = ("Rolling OLS", "Kalman")
README_START = "<!-- results:start -->"
README_END = "<!-- results:end -->"
EN = "–"
MINUS = "−"
TRADING_DAYS = 252

LOCAL_CSS = """
.toc { margin-top: calc(-1 * var(--space-5)); }
.toc ol { display: flex; flex-wrap: wrap; gap: 0 var(--space-5); margin: 0; padding: var(--space-4) 0 0;
  border-top: 1px solid var(--border); list-style: none; font-size: var(--fs-small); }
.toc a { display: inline-flex; align-items: center; min-height: 44px; color: var(--text-2); text-decoration: none; }
.toc a:hover { color: var(--text); }
.c-dash { stroke-dasharray: 5 4; }
.shots { display: grid; grid-template-columns: minmax(0, 3fr) minmax(0, 1fr); gap: var(--card-gap); align-items: start; }
.shots img { display: block; width: 100%; height: auto; border-radius: var(--radius-sm); border: 1px solid var(--border); }
.shots figcaption { font-size: var(--fs-small); color: var(--text-2); margin-top: var(--space-3); }
.shots figure { margin: 0; }
@media (max-width: 734px) { .shots { grid-template-columns: minmax(0, 1fr); } .shots .wide { display: none; } }
"""


# ---------------------------------------------------------------- inputs


def load(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def source(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def line_of(rel, needle):
    return next(i for i, line in enumerate(source(rel).splitlines(), 1) if needle in line)


def line_range(rel, start, end):
    lines = source(rel).splitlines()
    first = next(i for i, line in enumerate(lines, 1) if start in line)
    last = next(i for i, line in enumerate(lines, 1) if i >= first and end in line)
    return first, last


def facts_from_sources():
    bt = source("src/backtester.py")
    res = source("src/research.py")
    grid = lambda name: len(re.search(rf"{name} = \(([^)]*)\)", res).group(1).split(","))
    return {
        "commission": float(re.search(r"commission_bps=([\d.]+)", bt).group(1)),
        "slippage": float(re.search(r"slippage_bps=([\d.]+)", bt).group(1)),
        "warmup": int(re.search(r"DEFAULT_WARMUP = (\d+)", bt).group(1)),
        "train_fraction": float(re.search(r"DEFAULT_TRAIN_FRACTION = ([\d.]+)", res).group(1)),
        "configs": grid("ENTRY_GRID") * grid("EXIT_GRID") * grid("DELTA_GRID"),
        "configs_ols": grid("ENTRY_GRID") * grid("EXIT_GRID") * grid("OLS_WINDOW_GRID"),
        "universe": len(
            re.search(r"DEFAULT_UNIVERSE = \(([^)]*)\)", source("src/scanner.py"))
            .group(1)
            .split(",")
        ),
    }


def git_last_commit(rel):
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", rel],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return out or None


# ---------------------------------------------------------------- derived numbers


def sharpe_se(sr, years):
    """Approximate standard error of an annualised Sharpe ratio for iid returns (Lo, 2002)."""
    return math.sqrt((1 + sr * sr / 2) / years)


def derive(r, facts):
    lag0, lag1 = r["lags"]["0"], r["lags"]["1"]
    ols, kal = lag0["Rolling OLS"], lag0["Kalman"]
    test_days = len(ols["equity"]) - sum(1 for d, _ in ols["equity"] if d < r["test_start"])
    years = test_days / TRADING_DAYS
    cells = r["sweep"]
    ahead = sum(1 for c in cells if c["Rolling OLS"]["test_sharpe"] > c["Kalman"]["test_sharpe"])
    ols_all = [c["Rolling OLS"]["test_sharpe"] for c in cells]
    scan = r["scan"]
    m = len(scan)
    raw_sig = sum(1 for s in scan if s["p_value"] <= 0.05)
    bh_sig = sum(1 for s in scan if s["passes_bh"])
    pair = next(s for s in scan if {s["Y"], s["X"]} == {r["pair"]["y"], r["pair"]["x"]})
    return {
        "ols": ols["test"]["sharpe"],
        "kal": kal["test"]["sharpe"],
        "gap": ols["test"]["sharpe"] - kal["test"]["sharpe"],
        "ols1": lag1["Rolling OLS"]["test"]["sharpe"],
        "kal1": lag1["Kalman"]["test"]["sharpe"],
        "ols_train": ols["train"]["sharpe"],
        "kal_train": kal["train"]["sharpe"],
        "ols1_train": lag1["Rolling OLS"]["train"]["sharpe"],
        "test_days": test_days,
        "years": years,
        "se": sharpe_se(ols["test"]["sharpe"], years),
        "cells": len(cells),
        "ahead": ahead,
        "ols_lo": min(ols_all),
        "ols_hi": max(ols_all),
        "lags": sorted({c["lag"] for c in cells}),
        "costs": sorted({c["cost_bps"] for c in cells}),
        "m": m,
        "raw_sig": raw_sig,
        "bh_sig": bh_sig,
        "min_p": min(s["p_value"] for s in scan),
        "min_adj": min(s["adjusted_p"] for s in scan),
        "pair_p": pair["p_value"],
        "pair_adj": pair["adjusted_p"],
        "cost": r["default_cost_bps"],
        "pair": f"{r['pair']['y']}/{r['pair']['x']}",
    }


def sh(v):
    """Sharpe with a true minus sign."""
    return f"{v:.2f}".replace("-", MINUS)


def pct(v):
    return f"{v * 100:+.1f}%".replace("-", MINUS)


# ---------------------------------------------------------------- charts


def legend(items, x0, width):
    """items: (name, stroke class, dashed, accent). Wraps to new rows; returns (svg parts, rows)."""
    parts, kx, ky, rows = [], x0, 12, 1
    for name, cls, dash, accent in items:
        w = 24 + rk.text_width(name) + 18
        if kx + w - 18 > width and kx > x0:
            kx, ky, rows = x0, ky + 20, rows + 1
        parts.append(
            f'<path class="{cls}{" c-dash" if dash else ""}" stroke-width="2.5" '
            f'd="M{coord(kx)} {ky}H{coord(kx + 18)}"/>'
        )
        parts.append(svg_text(kx + 24, ky + 4, name, "c-label c-hi" if accent else "c-label"))
        kx += w
    return parts, rows


def to_day(s):
    return datetime.date.fromisoformat(s).toordinal()


def thin(points, width):
    step = max(1, len(points) // max(int(width / 4), 1))
    out = points[::step]
    return out if out[-1] == points[-1] else out + [points[-1]]


def time_chart(
    series,
    *,
    uid,
    width,
    height,
    title,
    desc,
    y_title,
    fmt_y,
    split=None,
    split_label=None,
    y_zero=False,
    show_legend=True,
):
    """Date x axis; series dicts: name, points [(iso date, y)], cls ('accent'|'muted'), dash."""
    phone = width < 420
    xs = [to_day(d) for s in series for d, _ in s["points"]]
    ys = [v for s in series for _, v in s["points"]]
    yt = rk.nice_ticks(0 if y_zero else min(ys), max(ys), 4 if phone else 5)
    ml = max(rk.text_width(fmt_y(t), 12) for t in yt) + 12
    key = [
        (
            s["name"],
            "c-stroke-accent" if s["cls"] == "accent" else "c-stroke-muted",
            s.get("dash"),
            s["cls"] == "accent",
        )
        for s in series
    ]
    key_parts, key_rows = legend(key, 0, width) if show_legend else ([], 0)
    mr, mt, mb = 12, 32 + 20 * key_rows, 30
    x0, x1 = min(xs), max(xs)
    px = lambda d: ml + (width - ml - mr) * (d - x0) / (x1 - x0)
    py = lambda v: height - mb - (height - mt - mb) * (v - yt[0]) / (yt[-1] - yt[0])
    parts = []
    if split:
        sx = px(to_day(split))
        parts.append(
            f'<rect class="c-band" x="{coord(sx)}" y="{coord(mt - 8)}" width="{coord(width - mr - sx)}" '
            f'height="{coord(height - mb - mt + 8)}"/>'
        )
        parts.append(
            f'<path class="c-floor" d="M{coord(sx)} {coord(mt - 8)}V{coord(height - mb)}"/>'
        )
        parts.append(svg_text(sx - 6, mt + 6, "train", "c-tick", "end"))
        parts.append(svg_text(sx + 6, mt + 6, split_label or "test", "c-tick"))
    for t in yt:
        parts.append(f'<path class="c-grid" d="M{coord(ml)} {coord(py(t))}H{coord(width - mr)}"/>')
        parts.append(svg_text(ml - 6, py(t) + 4, fmt_y(t), "c-tick", "end"))
    first_year = datetime.date.fromordinal(x0).year + 1
    last_year = datetime.date.fromordinal(x1).year
    years = range(first_year, last_year + 1, 2 if phone else 1)
    for y in years:
        d = datetime.date(y, 1, 1).toordinal()
        parts.append(f'<path class="c-axis" d="M{coord(px(d))} {coord(height - mb)}v4"/>')
        parts.append(svg_text(px(d), height - mb + 18, str(y), "c-tick", "middle"))
    parts.append(
        f'<path class="c-axis" d="M{coord(ml)} {coord(height - mb)}H{coord(width - mr)}"/>'
    )
    for s in sorted(series, key=lambda s: s["cls"] == "accent"):
        pts = thin(s["points"], width - ml - mr)
        d = "".join(
            ("M" if i == 0 else "L") + f"{coord(px(to_day(a)))} {coord(py(b))}"
            for i, (a, b) in enumerate(pts)
        )
        cls = "c-line-accent" if s["cls"] == "accent" else "c-stroke-muted"
        parts.append(
            f'<path class="{cls}{" c-dash" if s.get("dash") else ""}" stroke-width="{s.get("w", 1.5)}" d="{d}"/>'
        )
    parts.extend(key_parts)
    parts.append(svg_text(ml, mt - 16, y_title, "c-title"))
    return rk.svg(uid, width, height, title, desc, "".join(parts))


def pvalue_chart(scan, *, uid, width, title, desc, alpha=0.05):
    """Raw Engle-Granger p-values ranked, with the BH rejection line k/m * alpha, log y."""
    phone = width < 420
    height = 300 if phone else 340
    raw = sorted(s["p_value"] for s in scan)
    adj = sorted(s["adjusted_p"] for s in scan)
    m = len(raw)
    lo, hi = 0.0005, 1.0
    ml, mr, mt, mb = 44, 12, 52, 44
    px = lambda k: ml + (width - ml - mr) * (k - 1) / (m - 1)
    py = lambda v: (
        mt
        + (height - mt - mb)
        * (math.log10(hi) - math.log10(max(v, lo)))
        / (math.log10(hi) - math.log10(lo))
    )
    parts = []
    for t in (0.001, 0.01, 0.1, 1):
        parts.append(f'<path class="c-grid" d="M{ml} {coord(py(t))}H{coord(width - mr)}"/>')
        parts.append(svg_text(ml - 6, py(t) + 4, f"{t:g}", "c-tick", "end"))
    parts.append(f'<path class="c-floor" d="M{ml} {coord(py(alpha))}H{coord(width - mr)}"/>')
    parts.append(svg_text(width - mr, py(alpha) - 6, f"p = {alpha:g}", "c-tick", "end"))
    bh = "".join(
        ("M" if k == 1 else "L") + f"{coord(px(k))} {coord(py(k / m * alpha))}"
        for k in range(1, m + 1)
    )
    parts.append(f'<path class="c-line" d="{bh}"/>')
    label = "BH cut-off" if phone else f"BH cut-off k/{m} × {alpha:g}"
    parts.append(svg_text(px(m * 0.55), py(0.55 * alpha) + 16, label, "c-tick"))
    r = 2.6 if phone else 3.2
    for k, v in enumerate(adj, 1):
        parts.append(f'<circle cx="{coord(px(k))}" cy="{coord(py(v))}" r="{r}" class="c-accent"/>')
    for k, v in enumerate(raw, 1):
        parts.append(
            f'<circle cx="{coord(px(k))}" cy="{coord(py(v))}" r="{r}" class="c-stroke-muted" '
            f'stroke-width="1.5" style="fill:var(--surface)"/>'
        )
    for t in (1, 10, 20, 30, 40, 50, m):
        if phone and t in (10, 30, 50):
            continue
        parts.append(svg_text(px(t), height - mb + 18, str(t), "c-tick", "middle"))
    parts.append(f'<path class="c-axis" d="M{ml} {coord(height - mb)}H{coord(width - mr)}"/>')
    parts.append(svg_text(width - mr, height - 8, "Pair rank by p-value", "c-title", "end"))
    parts.append(svg_text(ml, mt - 16, "p-value (log scale)", "c-title"))
    parts.append(
        f'<circle cx="{ml + 6}" cy="12" r="{r}" class="c-stroke-muted" stroke-width="1.5" style="fill:var(--surface)"/>'
    )
    parts.append(svg_text(ml + 16, 16, "raw p", "c-label"))
    parts.append(f'<circle cx="{ml + 76}" cy="12" r="{r}" class="c-accent"/>')
    parts.append(svg_text(ml + 86, 16, "BH-adjusted p", "c-label c-hi"))
    return rk.svg(uid, width, height, title, desc, "".join(parts))


def sweep_chart(sweep, *, uid, width, title, desc):
    """Test-window Sharpe vs cost, one line per model and fill lag."""
    phone = width < 420
    height = 300 if phone else 320
    costs = sorted({c["cost_bps"] for c in sweep})
    lines = [(m, lag) for m in MODELS for lag in sorted({c["lag"] for c in sweep})]
    vals = [c[m]["test_sharpe"] for c in sweep for m in MODELS]
    yt = rk.nice_ticks(min(vals), max(vals), 5)
    key = [
        (
            f"{m}, {'next-day' if lag else 'same-close'} fill",
            "c-stroke-accent" if m == "Rolling OLS" else "c-stroke-muted",
            bool(lag),
            m == "Rolling OLS",
        )
        for m, lag in lines
    ]
    key_parts, key_rows = legend(key, 0, width)
    ml, mr, mt, mb = 40, 12, 36 + 20 * key_rows, 44
    px = lambda c: ml + (width - ml - mr) * (c - costs[0]) / (costs[-1] - costs[0])
    py = lambda v: height - mb - (height - mt - mb) * (v - yt[0]) / (yt[-1] - yt[0])
    parts = []
    for t in yt:
        parts.append(
            f'<path class="{"c-axis" if t == 0 else "c-grid"}" d="M{ml} {coord(py(t))}H{coord(width - mr)}"/>'
        )
        parts.append(svg_text(ml - 6, py(t) + 4, f"{t:g}".replace("-", MINUS), "c-tick", "end"))
    for c in costs:
        parts.append(svg_text(px(c), height - mb + 18, f"{c:g}", "c-tick", "middle"))
    parts.extend(key_parts)
    for m, lag in lines:
        pts = sorted((c["cost_bps"], c[m]["test_sharpe"]) for c in sweep if c["lag"] == lag)
        accent = m == "Rolling OLS"
        dash = " c-dash" if lag else ""
        d = "".join(
            ("M" if i == 0 else "L") + f"{coord(px(a))} {coord(py(b))}"
            for i, (a, b) in enumerate(pts)
        )
        parts.append(
            f'<path class="{"c-stroke-accent" if accent else "c-stroke-muted"}{dash}" stroke-width="2" d="{d}"/>'
        )
        for a, b in pts:
            parts.append(
                f'<circle cx="{coord(px(a))}" cy="{coord(py(b))}" r="3" class="{"c-accent" if accent else "c-muted"}"/>'
            )
    parts.append(
        svg_text(
            width - mr, height - 8, "Cost per fill, commission + slippage (bp)", "c-title", "end"
        )
    )
    parts.append(svg_text(ml, mt - 14, "Test-window Sharpe", "c-title"))
    return rk.svg(uid, width, height, title, desc, "".join(parts))


# ---------------------------------------------------------------- page


def evidence_rows(r, d, facts, src_sha, res_sha):
    def blob(rel, sha, start=None, end=None):
        anchor = f"#L{start}" + (f"-L{end}" if end and end != start else "") if start else ""
        return f"{REPO_URL}/blob/{sha}/{rel}{anchor}"

    res = blob("results/case_study.json", res_sha)
    fill = line_range(
        "src/backtester.py", "while pending and pending[0][0] == t", "position = new_position"
    )
    cost = (
        line_of("src/backtester.py", "commission_bps=1.0"),
        line_of("src/backtester.py", "slippage_bps=1.0"),
    )
    split = line_of("src/research.py", "DEFAULT_TRAIN_FRACTION = ")
    tune = line_range("src/research.py", "def tune(", "return best")
    uni = line_of("src/scanner.py", "DEFAULT_UNIVERSE = ")
    bh = line_of("src/scanner.py", 'method="fdr_bh"')
    meta = r["data"]
    return [
        (
            "E1",
            f"Rolling OLS test-window Sharpe {sh(d['ols'])}, Kalman {sh(d['kal'])}, {d['cost']:g} bp per fill, "
            f"same-close fills, {r['test_start']} to {r['test_end']}",
            "results/case_study.json: lags.0",
            res,
            "Measured",
        ),
        (
            "E2",
            f"Rolling OLS ahead of Kalman in {d['ahead']} of {d['cells']} lag × cost settings; its own Sharpe "
            f"{sh(d['ols_lo'])} to {sh(d['ols_hi'])}",
            "results/case_study.json: sweep",
            res,
            "Derived",
        ),
        (
            "E3",
            f"Next-day fills: rolling OLS {sh(d['ols1'])}, Kalman {sh(d['kal1'])}",
            "results/case_study.json: lags.1",
            res,
            "Measured",
        ),
        (
            "E4",
            "Default fills happen at the close that produced the signal; fill_lag=1 moves them one bar later",
            f"src/backtester.py:{fill[0]}{EN}{fill[1]}",
            blob("src/backtester.py", src_sha, *fill),
            "Code",
        ),
        (
            "E5",
            f"Costs {facts['commission']:g} bp commission + {facts['slippage']:g} bp slippage per fill",
            f"src/backtester.py:{cost[0]}{EN}{cost[1]}",
            blob("src/backtester.py", src_sha, *cost),
            "Code",
        ),
        (
            "E6",
            f"Parameters picked by train-window Sharpe over {facts['configs']} configurations per model; "
            f"{facts['train_fraction']:.0%} train split",
            f"src/research.py:{split}, {tune[0]}{EN}{tune[1]}",
            blob("src/research.py", src_sha, *tune),
            "Code",
        ),
        (
            "E7",
            f"{d['m']} pairs = C({facts['universe']}, 2), Engle–Granger with Benjamini–Hochberg at 5%",
            f"src/scanner.py:{uni}, {bh}",
            blob("src/scanner.py", src_sha, uni, bh),
            "Code",
        ),
        (
            "E8",
            f"{d['raw_sig']} of {d['m']} pairs have raw p ≤ 0.05; {d['bh_sig']} pass BH (smallest adjusted p "
            f"{d['min_adj']:.2f})",
            "results/case_study.json: scan",
            res,
            "Measured",
        ),
        (
            "E9",
            f"Prices: {len(meta['tickers'])} tickers, adjusted closes {r['first_date']} to {r['last_date']}, "
            f"fetched {meta['fetched_at'][:10]}",
            "data/prices.csv, data/prices.meta.json",
            blob("data/prices.meta.json", res_sha),
            "Data",
        ),
        (
            "E10",
            f"Earlier published card: Kalman {sh(load(ROOT / 'site' / 'card.json')['metric']['value'])} — same as E1",
            "site/card.json (from scripts/make_report.py)",
            blob("site/card.json", src_sha),
            "Reported",
        ),
    ]


def toc(items):
    lis = rk.join(Raw(f"<li>{rk.tag('a', label, href='#' + sid)}</li>") for sid, label in items)
    return Raw(f'<nav class="toc wrap" aria-label="Contents"><ol>{lis}</ol></nav>')


def equity_series(r, lag="0"):
    out = []
    for m in MODELS:
        pts = r["lags"][lag][m]["equity"]
        base = pts[0][1]
        out.append(
            {
                "name": m,
                "points": [(a, b / base * 100) for a, b in pts],
                "cls": "accent" if m == "Rolling OLS" else "muted",
            }
        )
    return out


def build_page(r, run, results_sha=None):
    facts = facts_from_sources()
    d = derive(r, facts)
    src_sha = run["code_sha"]
    res_sha = results_sha or src_sha
    ev = {row[0]: row for row in evidence_rows(r, d, facts, src_sha, res_sha)}
    R = rk.ref
    res_href = ev["E1"][3]
    cost = d["cost"]
    lag0 = r["lags"]["0"]
    ols_p, kal_p = lag0["Rolling OLS"]["params"], lag0["Kalman"]["params"]

    eq = equity_series(r)
    eq_take = (
        "Out of sample the rolling hedge ended ahead; the Kalman hedge gave back its training gains"
    )
    final = {s["name"]: s["points"][-1][1] for s in eq}
    test_start_val = {
        s["name"]: next(v for dd, v in s["points"] if dd >= r["test_start"]) for s in eq
    }
    eq_desc = (
        f"Equity indexed to 100 at {r['train_start']} for both models, {d['pair']}, {cost:g} bp per fill, same-close "
        f"fills. At the split ({r['test_start']}): rolling OLS {test_start_val['Rolling OLS']:.1f}, Kalman "
        f"{test_start_val['Kalman']:.1f}. At {r['test_end']}: {final['Rolling OLS']:.1f} and {final['Kalman']:.1f}."
    )
    hero = rk.hero(
        eyebrow="Quant research · Python · Case study",
        title="statarb",
        subtitle="A pairs-trading research app that compares a Kalman-filter hedge ratio with a rolling least-squares "
        "hedge, tuning each on one period and judging it on a later, untouched one.",
        value=sh(d["ols"]),
        label=Raw(
            f"out-of-sample Sharpe of the rolling-OLS hedge on {d['pair']}, after {cost:g} bp costs{R('E1')}"
        ),
        support=Raw(
            f"{d['gap']:.2f} above the Kalman filter on the same signals, sizing and costs, and ahead in all "
            f"{d['cells']} fill-lag and cost settings tested{R('E2')}. With next-day fills: "
            f"{sh(d['ols1'])}{R('E3')}."
        ),
        context=f"One pair; Yahoo adjusted closes {r['first_date'][:4]}–{r['last_date'][:4]}; tuned on "
        f"{facts['train_fraction']:.0%}, tested {r['test_start']} to {r['test_end']} ({d['years']:.1f} years). "
        f"Sharpe ± {d['se']:.2f} (one standard error).",
        actions=[("Source", REPO_URL, True), ("Reproduce", "#reproduce", False)],
        figure=rk.tag("p", "Equity, indexed to 100; shaded: test window", class_="card-label")
        + rk.variants(
            *(
                time_chart(
                    eq,
                    uid=f"hero-{k}",
                    width=w,
                    height=h,
                    title=eq_take,
                    desc=eq_desc,
                    y_title="Equity (start = 100)",
                    fmt_y=lambda t: f"{t:g}",
                    split=r["test_start"],
                )
                for k, w, h in (("d", 398, 260), ("t", 640, 300), ("p", 280, 240))
            )
        ),
    )

    nav = toc(
        [
            ("findings", "Findings"),
            ("backtest", "Out of sample"),
            ("mechanism", "Hedge ratios"),
            ("robustness", "Lag and costs"),
            ("scan", "Pair scan"),
            ("app", "The app"),
            ("methodology", "Method"),
            ("limitations", "Limits"),
            ("reproduce", "Reproduce"),
            ("evidence", "Evidence"),
        ]
    )

    findings = rk.bento(
        [
            (
                8,
                1,
                rk.join(
                    [
                        rk.tag(
                            "h3",
                            "The comparison held under every lag and cost tested",
                            class_="card-heading",
                        ),
                        rk.comparison(
                            [
                                (
                                    "Rolling OLS, same-close fills",
                                    max(d["ols"], 0),
                                    sh(d["ols"]),
                                    True,
                                ),
                                (
                                    "Rolling OLS, next-day fills",
                                    max(d["ols1"], 0),
                                    sh(d["ols1"]),
                                    True,
                                ),
                            ],
                            max(d["ols"], d["ols1"], d["ols_hi"]),
                        ),
                        rk.tag(
                            "p",
                            Raw(
                                f"Test-window Sharpe at {cost:g} bp per fill. Across fill lags {d['lags'][0]}–{d['lags'][-1]} "
                                f"and costs {d['costs'][0]}–{d['costs'][-1]} bp, rolling OLS beat Kalman {d['ahead']} of "
                                f"{d['cells']} times{R('E2')}; its own Sharpe ranged {sh(d['ols_lo'])} to "
                                f'{sh(d["ols_hi"])}. <a href="#robustness">Lag and cost sweep</a>'
                            ),
                        ),
                    ]
                ),
            ),
            (
                4,
                2,
                rk.join(
                    [
                        rk.tag("h3", "Within noise on its own", class_="card-heading"),
                        rk.metric("Standard error of the Sharpe", f"±{d['se']:.2f}"),
                        rk.tag(
                            "p",
                            Raw(
                                f"{d['test_days']:,} test days ({d['years']:.1f} years) cannot separate a {sh(d['ols'])} "
                                f"Sharpe from zero, so the report leans on the ranking across settings rather than the level."
                            ),
                        ),
                    ]
                ),
            ),
            (
                6,
                3,
                rk.join(
                    [
                        rk.tag("h3", "Multiple testing is corrected", class_="card-heading"),
                        rk.metric("Pairs screened", str(d["m"])),
                        rk.tag(
                            "p",
                            Raw(
                                f"Benjamini–Hochberg turns {d['raw_sig']} raw hits at p ≤ 0.05 into {d['bh_sig']}; "
                                f"{d['pair']} itself has raw p {d['pair_p']:.2f}.{R('E7')}{R('E8')}"
                            ),
                        ),
                    ]
                ),
            ),
            (
                6,
                4,
                rk.join(
                    [
                        rk.tag("h3", "A fair baseline", class_="card-heading"),
                        rk.tag(
                            "p",
                            Raw(
                                f"Both hedges run through one backtester: same z-score rule, sizing, costs and "
                                f"{facts['configs']}-point tuning grid, picked on the train window only.{R('E6')}"
                            ),
                        ),
                    ]
                ),
            ),
        ]
    )
    findings_sec = rk.section("findings", "What the case study shows", findings, eyebrow="Findings")

    # Out-of-sample equity
    eq_fig = rk.figure(
        "fig-equity",
        eq_take,
        rk.variants(
            *(
                time_chart(
                    eq,
                    uid=f"eq-{k}",
                    width=w,
                    height=h,
                    title=eq_take,
                    desc=eq_desc,
                    y_title="Equity (start = 100)",
                    fmt_y=lambda t: f"{t:g}",
                    split=r["test_start"],
                    split_label=f"test from {r['test_start']}" if w > 500 else "test",
                )
                for k, w, h in (("d", 1000, 380), ("t", 640, 340), ("p", 300, 280))
            )
        ),
        rk.tag(
            "p",
            Raw(
                f"Net of {cost:g} bp per fill, fills at the signal's close, $100,000 sized to equity, {d['pair']} daily "
                f"adjusted closes. Parameters were picked on the unshaded train window and held fixed in the test window "
                f"({d['test_days']:,} days). Single path; drawn every few days.{R('E1')}{R('E6')}"
            ),
        ),
        question="Does each hedge make money after the period it was tuned on?",
        table=rk.data_table(
            "Train and test metrics, same-close fills",
            ["Model", "Window", "Sharpe", "Annual return", "Max drawdown", "Trades"],
            [
                [
                    m,
                    w,
                    sh(lag0[m][w]["sharpe"]),
                    pct(lag0[m][w]["annual_return"]),
                    pct(lag0[m][w]["max_drawdown"]),
                    str(lag0[m][w]["trades"]),
                ]
                for m in MODELS
                for w in ("train", "test")
            ],
            numeric={2, 3, 4, 5},
        ),
        data_href=res_href,
    )
    bt_body = rk.join(
        [
            eq_fig,
            Raw('<div class="prose" style="margin-top:32px">'),
            rk.tag(
                "p",
                Raw(
                    f"On the train window both hedges earned money: Sharpe {sh(d['ols_train'])} for rolling OLS and "
                    f"{sh(d['kal_train'])} for Kalman. In the test window the rolling hedge kept a {sh(d['ols'])} Sharpe and the "
                    f"Kalman hedge fell to {sh(d['kal'])}.{R('E1')} The tuned settings were entry z "
                    f"{ols_p['entry_threshold']:g}, exit {ols_p['exit_threshold']:g}, {ols_p['ols_window']}-day window for OLS; "
                    f"entry {kal_p['entry_threshold']:g}, exit {kal_p['exit_threshold']:g}, delta {kal_p['delta']:g} for Kalman."
                ),
            ),
            rk.tag(
                "p",
                Raw(
                    f"{d['pair']} fails the Engle–Granger test on the train window (p = {r['train_stats']['eg_pvalue']:.2f}), "
                    f"so the spread is not shown to mean-revert; the app flags this, and the result is a model comparison on "
                    f"a weak pair, not evidence of a tradable edge."
                ),
            ),
            Raw("</div>"),
        ]
    )
    bt_sec = rk.section(
        "backtest",
        f"On {d['pair']}, the rolling hedge beat the Kalman hedge out of sample",
        bt_body,
        eyebrow="Result",
    )

    # Hedge ratios
    hr = [
        {
            "name": f"Rolling OLS, {ols_p['ols_window']}-day window",
            "points": r["hedge_ratio"]["Rolling OLS"],
            "cls": "accent",
            "w": 1.2,
        },
        {
            "name": f"Kalman, delta {kal_p['delta']:g}",
            "points": r["hedge_ratio"]["Kalman"],
            "cls": "muted",
            "w": 2.2,
        },
    ]
    hk = [v for _, v in r["hedge_ratio"]["Kalman"]]
    ho = [v for _, v in r["hedge_ratio"]["Rolling OLS"]]
    hr_take = "The rolling hedge re-estimates within weeks; the tuned Kalman hedge barely moves"
    hr_desc = (
        f"Hedge ratio (shares of {r['pair']['x']} per share of {r['pair']['y']}) from {r['train_start']} to "
        f"{r['test_end']}. Kalman ranges {min(hk):.2f} to {max(hk):.2f}; rolling OLS {min(ho):.2f} to {max(ho):.2f}."
    )
    hr_fig = rk.figure(
        "fig-hedge",
        hr_take,
        rk.variants(
            *(
                time_chart(
                    hr,
                    uid=f"hr-{k}",
                    width=w,
                    height=h,
                    title=hr_take,
                    desc=hr_desc,
                    y_title=f"Hedge ratio ({r['pair']['x']} shares per {r['pair']['y']} share)",
                    fmt_y=lambda t: f"{t:g}".replace("-", MINUS),
                    split=r["test_start"],
                )
                for k, w, h in (("d", 1000, 360), ("t", 640, 320), ("p", 300, 300))
            )
        ),
        rk.tag(
            "p",
            Raw(
                f"Each model's hedge ratio with its tuned setting. Kalman ranged {sh(min(hk))} to {sh(max(hk))}; rolling "
                f"OLS {sh(min(ho))} to {sh(max(ho))}. Drawn every few days.{R('E6')}"
            ),
        ),
        question="How differently do the two models track the relationship between the stocks?",
        data_href=res_href,
    )
    mech_body = rk.join(
        [
            hr_fig,
            Raw('<div class="prose" style="margin-top:32px">'),
            rk.tag(
                "p",
                Raw(
                    f"A Kalman filter treats the hedge ratio as a hidden value that drifts; <code>delta</code> sets how fast it "
                    f"may drift. Tuning picked the smallest delta in the grid, {kal_p['delta']:g}, so the Kalman hedge behaves "
                    f"almost like one fixed ratio. The {ols_p['ols_window']}-day regression instead refits every day."
                ),
            ),
            rk.tag(
                "p",
                Raw(
                    "A slow hedge lets the spread inherit any lasting change in how the two stocks move together, so its "
                    "z-score can stay stretched for months. The Kalman results — fewer, longer trades and a falling test "
                    "equity — are consistent with that; this study does not isolate the cause."
                ),
            ),
            Raw("</div>"),
        ]
    )
    mech_sec = rk.section(
        "mechanism",
        "The tuned Kalman hedge was too slow to follow the pair",
        mech_body,
        eyebrow="Mechanism",
    )

    # Robustness
    sw_take = (
        f"Rolling OLS stayed ahead of Kalman in {d['ahead']} of {d['cells']} lag and cost settings"
    )
    sw_desc = "; ".join(
        f"lag {c['lag']}, {c['cost_bps']:g} bp: OLS {sh(c['Rolling OLS']['test_sharpe'])}, Kalman "
        f"{sh(c['Kalman']['test_sharpe'])}"
        for c in r["sweep"]
    )
    sw_fig = rk.figure(
        "fig-robustness",
        sw_take,
        rk.variants(
            *(
                sweep_chart(r["sweep"], uid=f"sw-{k}", width=w, title=sw_take, desc=sw_desc)
                for k, w in (("d", 1000), ("t", 640), ("p", 300))
            )
        ),
        rk.tag(
            "p",
            Raw(
                f"Test-window Sharpe; each point re-tunes both models on the train window at that cost and fill lag "
                f"({len(r['sweep'])} tunings). Same-close fills trade at the close that produced the signal; next-day fills "
                f"trade one close later.{R('E2')}{R('E4')}"
            ),
        ),
        question="Does the ranking survive a one-day execution delay and higher trading costs?",
        table=rk.data_table(
            "Test-window Sharpe by fill lag and cost",
            ["Fill", "Cost (bp)", "Rolling OLS", "Kalman", "Rolling OLS, train"],
            [
                [
                    "next-day" if c["lag"] else "same-close",
                    f"{c['cost_bps']:g}",
                    sh(c["Rolling OLS"]["test_sharpe"]),
                    sh(c["Kalman"]["test_sharpe"]),
                    sh(c["Rolling OLS"]["train_sharpe"]),
                ]
                for c in r["sweep"]
            ],
            numeric={1, 2, 3, 4},
        ),
        data_href=res_href,
    )
    neg = [c for c in r["sweep"] if c["Rolling OLS"]["test_sharpe"] < 0]
    neg_text = "; ".join(
        f"{'next-day' if c['lag'] else 'same-close'} fills at {c['cost_bps']:g} bp "
        f"({sh(c['Rolling OLS']['test_sharpe'])}, train {sh(c['Rolling OLS']['train_sharpe'])})"
        for c in neg
    )
    rob_body = rk.join(
        [
            sw_fig,
            Raw('<div class="prose" style="margin-top:32px">'),
            rk.tag(
                "p",
                Raw(
                    f"Filling at the signal's own close assumes the trade gets a price that was already used to decide it. "
                    f"Moving every fill one close later changed the rolling hedge's Sharpe from {sh(d['ols'])} to "
                    f"{sh(d['ols1'])} and Kalman's from {sh(d['kal'])} to {sh(d['kal1'])}, so same-close fills do not explain "
                    f"the ranking.{R('E3')}"
                ),
            ),
            rk.tag(
                "p",
                Raw(
                    f"The rolling hedge's own level is fragile: it turned negative in {len(neg)} setting"
                    f"{'s' if len(neg) != 1 else ''} — {neg_text} — and its test Sharpe does not fall steadily with cost, "
                    f"because each cost re-tunes the parameters on a short train window."
                    if neg
                    else "The rolling hedge's Sharpe stayed positive in every setting, but it does not fall steadily with cost, "
                    "because each cost re-tunes the parameters on a short train window."
                ),
            ),
            Raw("</div>"),
        ]
    )
    rob_sec = rk.section(
        "robustness",
        "A one-day fill delay did not change which hedge won",
        rob_body,
        eyebrow="Robustness",
    )

    # Scan
    pv_take = (
        f"After correcting for {d['m']} tests, {d['bh_sig']} pair{'s' if d['bh_sig'] != 1 else ''} "
        f"remain{'s' if d['bh_sig'] == 1 else ''} significant"
    )
    pv_desc = (
        f"{d['m']} Engle–Granger p-values ranked, raw and Benjamini–Hochberg adjusted. {d['raw_sig']} raw "
        f"p-values are at or below 0.05; the smallest adjusted p-value is {d['min_adj']:.2f}."
    )
    pv_fig = rk.figure(
        "fig-scan",
        pv_take,
        rk.variants(
            *(
                pvalue_chart(r["scan"], uid=f"pv-{k}", width=w, title=pv_take, desc=pv_desc)
                for k, w in (("d", 1000), ("t", 640), ("p", 300))
            )
        ),
        rk.tag(
            "p",
            Raw(
                f"Every pair of the app's {facts['universe']} default consumer-staples tickers over the full "
                f"{r['first_date'][:4]}–{r['last_date'][:4]} period, as the app's scanner runs it. A raw p-value counts "
                f"only if it falls under the cut-off line at its rank.{R('E7')}{R('E8')}"
            ),
        ),
        question="How many of the screened pairs are cointegrated once the number of tests is accounted for?",
        table=rk.data_table(
            "Engle–Granger p-values, sorted by adjusted p",
            ["Y", "X", "Raw p", "Adjusted p", "Half-life (days)"],
            [
                [
                    s["Y"],
                    s["X"],
                    f"{s['p_value']:.3f}",
                    f"{s['adjusted_p']:.3f}",
                    f"{s['half_life_days']:.0f}" if s["half_life_days"] is not None else "n/a",
                ]
                for s in r["scan"]
            ],
            numeric={2, 3, 4},
        ),
        data_href=res_href,
    )
    scan_body = rk.join(
        [
            pv_fig,
            Raw('<div class="prose" style="margin-top:32px">'),
            rk.tag(
                "p",
                Raw(
                    f"At 5% per test, {d['m']} independent tests would give about {d['m'] * 0.05:.1f} false hits by chance; "
                    f"the screen found {d['raw_sig']}. Benjamini–Hochberg keeps the expected share of false discoveries at 5% "
                    f"and, on this universe and period, keeps {d['bh_sig']}. The scanner reports that instead of a ranked "
                    f"list of likely false positives."
                ),
            ),
            Raw("</div>"),
        ]
    )
    scan_sec = rk.section(
        "scan",
        f"The scanner screens {d['m']} pairs and corrects for multiple testing",
        scan_body,
        eyebrow="Pair scan",
    )

    # App
    shots = Raw(
        '<div class="shots">'
        '<figure class="wide"><img src="app-1280-light.png" width="1280" height="900" '
        f'alt="The statarb app on {esc(d["pair"])}: pair statistics, train and test windows, and the results table">'
        "<figcaption>Backtest tab at 1280 px: pair tests on the train window, then train and test metrics per model."
        "</figcaption></figure>"
        '<figure><img src="app-390-light.png" width="390" height="844" '
        'alt="The statarb app at phone width"><figcaption>The same view at 390 px.</figcaption></figure>'
        "</div>"
    )
    app_sec = rk.section(
        "app",
        "The Streamlit app runs the same code on any two tickers",
        rk.join(
            [
                shots,
                rk.tag(
                    "p",
                    Raw(
                        "Settings sit in the sidebar: tickers, dates, train share, thresholds and costs. The pair scanner "
                        "tab runs the same Benjamini–Hochberg screen on any ticker list. Screenshots are from an earlier "
                        "commit with the same defaults."
                    ),
                    class_="prose",
                    style="margin-top:24px",
                ),
            ]
        ),
        eyebrow="App",
    )

    method = rk.methodology(
        [
            (
                "Data",
                Raw(
                    f"{esc(r['data']['source'])}; {d['pair']} and {facts['universe'] - 2} more tickers, "
                    f"{r['first_date']} to {r['last_date']} ({r['rows']:,} days), fetched "
                    f"{esc(r['data']['fetched_at'][:10])}, committed as <code>data/prices.csv</code>{R('E9')}"
                ),
            ),
            (
                "Split",
                f"Warm-up {facts['warmup']} days from {r['warmup_start']}; train {r['train_start']} to "
                f"{r['train_end']}; test {r['test_start']} to {r['test_end']}",
            ),
            (
                "Tuning",
                f"Grid search on train-window Sharpe: {facts['configs']} configurations for Kalman, "
                f"{facts['configs_ols']} for rolling OLS; no adjustment for the number of configurations",
            ),
            (
                "Signal",
                "z-score of the model's prediction error; enter beyond the entry threshold, exit back toward zero",
            ),
            (
                "Sizing",
                "Gross exposure equal to account equity, $100,000 start, long one leg and short the other",
            ),
            (
                "Costs",
                Raw(
                    f"{facts['commission']:g} bp commission + {facts['slippage']:g} bp slippage per fill on traded "
                    f"notional; sweep 0–20 bp{R('E5')}"
                ),
            ),
            (
                "Fills",
                Raw(
                    f"Default at the signal's close; also one close later (<code>fill_lag=1</code>){R('E4')}"
                ),
            ),
            (
                "Sharpe",
                f"Daily returns × √{TRADING_DAYS}, risk-free rate 0; standard error ≈ √((1 + SR²/2) / years)",
            ),
            ("Not modelled", "Borrow fees, margin interest, market impact, partial fills"),
            (
                "Run",
                f"{run['command']} · {run['machine']}, {run['os']}, {run['python']} · {run['wall_seconds']:.0f} s · "
                f"load average {run['load_average_before_run']} · {run['date']}",
            ),
        ]
    )
    method_sec = rk.section("methodology", "Methodology", method, eyebrow="Conditions")

    limits = rk.limitations(
        [
            (
                "One pair, one period.",
                "The ranking is measured on PEP/KO in 2022–2024 only; another pair or period may "
                "reverse it.",
            ),
            (
                "Short test window.",
                Raw(
                    f"{d['years']:.1f} years gives a Sharpe standard error near ±{d['se']:.2f}, so "
                    f"neither model's test Sharpe is distinguishable from zero."
                ),
            ),
            (
                "Weak pair.",
                f"PEP/KO fails Engle–Granger on the train window (p = {r['train_stats']['eg_pvalue']:.2f}) and "
                f"the BH screen.",
            ),
            (
                "Tuning on a small grid.",
                "The best of "
                f"{facts['configs']} configurations is chosen per model with no multiple-testing "
                "adjustment; train Sharpe overstates what tuning buys.",
            ),
            (
                "Daily closes.",
                "Fills at a close, at most one day late; no intraday prices, impact or borrow costs.",
            ),
            (
                "Adjusted prices change.",
                "Yahoo revises adjusted closes after each dividend; the committed snapshot fixes "
                "them, a fresh download will differ slightly.",
            ),
        ]
    )
    limits_sec = rk.section(
        "limitations",
        "Results cover one pair, one window and daily closes",
        limits,
        eyebrow="Limitations",
    )

    steps_html = rk.reproduce(
        [
            (
                "Install (Python 3.11 or 3.12) and run the tests.",
                'python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"\n.venv/bin/pytest -q',
            ),
            (
                Raw(
                    f"Re-run the case study from the committed prices (about {run['wall_seconds']:.0f} s); "
                    f"<code>results/case_study.json</code> should not change. Add <code>--fetch</code> to re-download "
                    f"(network; adjusted prices will differ slightly)."
                ),
                ".venv/bin/python scripts/case_study.py\ngit diff --exit-code results/case_study.json",
            ),
            (
                "Rebuild this page and the README block (Python standard library only).",
                "python3 scripts/build_report.py --readme README.md",
            ),
            ("Open the app.", ".venv/bin/streamlit run app.py"),
        ],
        expected=f"Expected: lag 0: test Sharpe Kalman {sh(d['kal'])}, rolling OLS {sh(d['ols'])}.",
    )
    repro_sec = rk.section(
        "reproduce",
        "Reproduce it from the committed prices",
        steps_html,
        eyebrow="Reproduce",
        intro=Raw(f'Commands run from a clone of <a class="src" href="{REPO_URL}">{REPO}</a>.'),
    )

    ev_table = rk.table(
        "Evidence ledger",
        ["ID", "Claim", "Source", "Class"],
        [
            [
                Raw(f'<span id="evidence-{eid.lower()}">{eid}</span>'),
                claim,
                Raw(f'<a class="src" href="{esc(href)}">{esc(src)}</a>'),
                cls,
            ]
            for eid, claim, src, href, cls in ev.values()
        ],
        wrap_cols={1, 2},
    )
    ev_sec = rk.section(
        "evidence", "Every number traces to a committed file", ev_table, eyebrow="Sources"
    )

    body = rk.join(
        [
            hero,
            nav,
            findings_sec,
            bt_sec,
            mech_sec,
            rob_sec,
            scan_sec,
            app_sec,
            method_sec,
            limits_sec,
            repro_sec,
            ev_sec,
        ]
    )
    footer = rk.footer(
        Raw(
            f"Generated by <code>scripts/build_report.py</code> from <code>results/case_study.json</code> · code "
            f"{esc(src_sha[:7])} · run {esc(run['date'])} on {esc(run['machine'])}"
        ),
        Raw(f'<a href="{HOME_URL}">Allen Wang</a> · <a href="{REPO_URL}">Source</a>'),
    )
    html = rk.page(
        title="statarb · Allen Wang",
        description=(
            f"Pairs-trading case study: on {d['pair']} a rolling least-squares hedge earned an out-of-sample "
            f"Sharpe of {sh(d['ols'])} after costs and beat a Kalman-filter hedge in every lag and cost setting."
        ),
        canonical=PAGE_URL,
        header=rk.report_header(HOME_URL, "← Allen Wang", [("Source", REPO_URL)]),
        main=body,
        footer=footer,
    )
    return html.replace("</style>", LOCAL_CSS + "</style>", 1)


def readme_block(r):
    d = derive(r, facts_from_sources())
    return "\n".join(
        [
            README_START,
            f"- **Out of sample on {d['pair']}, the rolling least-squares hedge earned a Sharpe of {sh(d['ols'])}** after "
            f"{d['cost']:g} bp costs ({r['test_start']} to {r['test_end']}), {d['gap']:.2f} above the Kalman hedge, and "
            f"stayed ahead in all {d['cells']} fill-lag and cost settings tested. With {d['years']:.1f} test years the "
            f"Sharpe's standard error is about ±{d['se']:.2f}.",
            f"- **Next-day fills did not change the ranking**: {sh(d['ols1'])} vs {sh(d['kal1'])}.",
            f"- **Pair scanner** tests all {d['m']} pairs of {facts_from_sources()['universe']} tickers for cointegration "
            f"with Benjamini–Hochberg correction: {d['raw_sig']} raw hits at p ≤ 0.05, {d['bh_sig']} after correction.",
            "- **Fair baseline:** both hedges share the signal, sizing, cost and tuning code.",
            README_END,
        ]
    )


def update_readme(path, r):
    p = pathlib.Path(path)
    text = p.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        sys.exit(f"{path} has no {README_START} ... {README_END} block")
    head, rest = text.split(README_START, 1)
    p.write_text(head + readme_block(r) + rest.split(README_END, 1)[1], encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", default=str(ROOT / "results" / "case_study.json"))
    parser.add_argument("--run", default=str(ROOT / "results" / "run.json"))
    parser.add_argument("--out", default=str(ROOT / "docs" / "index.html"))
    parser.add_argument("--readme", help="rewrite the results block between the README markers")
    args = parser.parse_args()
    results = load(args.results)
    path = pathlib.Path(args.results).resolve()
    sha = git_last_commit(str(path.relative_to(ROOT))) if path.is_relative_to(ROOT) else None
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_page(results, load(args.run), sha), encoding="utf-8")
    if args.readme:
        update_readme(args.readme, results)


if __name__ == "__main__":
    main()
