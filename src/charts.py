"""Plotly figures shared by the Streamlit app and the static report.

Colors are Okabe-Ito (colorblind-safe); every pair of series also differs by
dash or marker shape. No template is set, so Streamlit applies its own light
or dark theme.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

from src.metrics import drawdown

BLUE = "#0072B2"  # Kalman, and Y in the price panel
VERMILLION = "#D55E00"  # Rolling OLS, and X in the price panel
GREEN = "#009E73"  # long-spread entries
PURPLE = "#CC79A7"  # short-spread entries
GREY = "#7F7F7F"  # thresholds, exits, reference lines
SHADE = "rgba(127, 127, 127, 0.15)"
ACF_LAGS = 20
Z_95 = 1.96  # two-sided 95% normal quantile

MODEL_STYLE = {
    "Kalman": dict(color=BLUE, dash="solid"),
    "Rolling OLS": dict(color=VERMILLION, dash="dash"),
}


def _layout(fig, height, y_title, **kwargs):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        **kwargs,
    )
    fig.update_yaxes(title_text=y_title)
    return fig


def _shade_windows(fig, ev, rows=None):
    kw = {} if rows is None else dict(row="all", col=1)
    fig.add_vrect(
        x0=ev.warmup_start,
        x1=ev.train_start,
        fillcolor=SHADE,
        line_width=0,
        annotation_text="warm-up",
        annotation_position="top left",
        **kw,
    )
    fig.add_vrect(
        x0=ev.test_start,
        x1=ev.test_end,
        fillcolor=SHADE,
        line_width=0,
        annotation_text="test window",
        annotation_position="top left",
        **kw,
    )


def prices_and_spread(data, result, ev, ticker_y, ticker_x):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.6, 0.4]
    )
    for col, name, color, dash in (
        ("asset_y", f"{ticker_y} (Y)", BLUE, "solid"),
        ("asset_x", f"{ticker_x} (X)", VERMILLION, "dash"),
    ):
        rebased = 100 * data[col] / data[col].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=data.index, y=rebased, name=name, line=dict(color=color, dash=dash, width=2)
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=result.index,
            y=result["innovation"],
            name="Spread (Kalman innovation)",
            line=dict(color=GREY, width=1),
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Price (rebased to 100)", row=1, col=1)
    fig.update_yaxes(title_text="Spread y - (a + b x), $", row=2, col=1)
    _shade_windows(fig, ev, rows=True)
    return _layout(fig, 520, None)


def zscore_chart(result, entry, exit_, model_name):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.index,
            y=result["z_score"],
            name=f"z ({model_name})",
            line=dict(color=MODEL_STYLE[model_name]["color"], width=1),
        )
    )
    for level, label in ((entry, "entry"), (-entry, None), (exit_, "exit"), (-exit_, None)):
        fig.add_hline(
            y=level, line=dict(color=GREY, dash="dash" if label != "exit" else "dot", width=1)
        )
    pos = result["position"]
    prev = pos.shift(1).fillna(0)
    z = result["z_score"]
    for mask, name, symbol, color in (
        ((prev == 0) & (pos == 1), "Enter long spread", "triangle-up", GREEN),
        ((prev == 0) & (pos == -1), "Enter short spread", "triangle-down", PURPLE),
        ((prev != 0) & (pos == 0), "Exit", "circle", GREY),
    ):
        fig.add_trace(
            go.Scatter(
                x=result.index[mask],
                y=z[mask],
                mode="markers",
                name=name,
                marker=dict(symbol=symbol, size=10, color=color),
            )
        )
    return _layout(fig, 360, "z (innovation / sqrt(S))")


def hedge_ratio_chart(kalman, ols, ticker_y, ticker_x):
    band_hi = kalman["hedge_ratio"] + 2 * kalman["beta_sd"]
    band_lo = kalman["hedge_ratio"] - 2 * kalman["beta_sd"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=kalman.index, y=band_hi, line=dict(width=0), showlegend=False, hoverinfo="skip"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=kalman.index,
            y=band_lo,
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(0, 114, 178, 0.2)",
            name="Kalman +/- 2 sd",
        )
    )
    for name, res in (("Kalman", kalman), ("Rolling OLS", ols)):
        fig.add_trace(
            go.Scatter(
                x=res.index,
                y=res["hedge_ratio"],
                name=f"{name} beta",
                line=dict(width=2, **MODEL_STYLE[name]),
            )
        )
    return _layout(fig, 360, f"beta (shares {ticker_x} per share {ticker_y})")


def equity_and_drawdown(runs, ev):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.65, 0.35]
    )
    for name, run in runs.items():
        equity = run.result["equity"]
        style = MODEL_STYLE[name]
        fig.add_trace(
            go.Scatter(
                x=equity.index,
                y=equity,
                name=name,
                legendgroup=name,
                line=dict(width=2, **style),
                hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=equity.index,
                y=100 * drawdown(equity),
                name=name,
                legendgroup=name,
                showlegend=False,
                line=dict(width=1, **style),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}%",
            ),
            row=2,
            col=1,
        )
    fig.update_yaxes(title_text="Equity, $ (net of costs)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown, %", row=2, col=1)
    _shade_windows(fig, ev, rows=True)
    return _layout(fig, 520, None)


def autocorrelation(values, lags=ACF_LAGS):
    v = np.asarray(values, dtype=float)
    v = v - v.mean()
    denom = np.dot(v, v)
    return np.array([np.dot(v[:-k], v[k:]) / denom for k in range(1, lags + 1)])


def innovation_diagnostics(result):
    z = result["z_score"].dropna().to_numpy()
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.12,
        subplot_titles=(
            "Standardized innovations vs N(0,1)",
            f"Autocorrelation, lags 1-{ACF_LAGS}",
        ),
    )
    fig.add_trace(
        go.Histogram(x=z, histnorm="probability density", name="z", marker_color=BLUE, opacity=0.7),
        row=1,
        col=1,
    )
    grid = np.linspace(-4, 4, 200)
    fig.add_trace(
        go.Scatter(x=grid, y=norm.pdf(grid), name="N(0,1)", line=dict(color=GREY, dash="dash")),
        row=1,
        col=1,
    )
    acf = autocorrelation(z)
    fig.add_trace(
        go.Bar(x=list(range(1, ACF_LAGS + 1)), y=acf, name="ACF", marker_color=BLUE), row=1, col=2
    )
    band = Z_95 / np.sqrt(len(z))
    for level in (band, -band):
        fig.add_hline(y=level, line=dict(color=GREY, dash="dash", width=1), row=1, col=2)
    fig.update_xaxes(title_text="z", row=1, col=1)
    fig.update_xaxes(title_text="lag (days)", row=1, col=2)
    fig.update_yaxes(title_text="density", row=1, col=1)
    fig.update_yaxes(title_text="autocorrelation", row=1, col=2)
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=60, b=10), showlegend=False)
    return fig


def diagnostics_takeaway(result):
    z = result["z_score"].dropna().to_numpy()
    acf = autocorrelation(z)
    outside = int((np.abs(acf) > Z_95 / np.sqrt(len(z))).sum())
    return (
        f"Standardized innovations have variance {np.var(z):.2f} (1.00 if the model fits) "
        f"and {outside} of {ACF_LAGS} autocorrelations outside the 95% band."
    )
