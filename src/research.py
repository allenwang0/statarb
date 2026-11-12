"""Train/test evaluation: tune on the train window, report on the test window."""

from dataclasses import dataclass, field
from itertools import product

import pandas as pd

from src.backtester import DEFAULT_WARMUP, Backtester
from src.data import DataError
from src.metrics import compute_metrics
from src.stats import PairStats, pair_stats

ENTRY_GRID = (1.0, 1.5, 2.0, 2.5)
EXIT_GRID = (0.0, 0.5)
DELTA_GRID = (1e-5, 1e-4, 1e-3)
OLS_WINDOW_GRID = (20, 40, 60)
DEFAULT_TRAIN_FRACTION = 0.6


@dataclass
class ModelRun:
    params: dict
    result: pd.DataFrame
    train: dict
    test: dict


@dataclass
class Evaluation:
    warmup_start: pd.Timestamp
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    stats: PairStats  # on the train window only
    runs: dict = field(default_factory=dict)  # 'Kalman' / 'Rolling OLS' -> ModelRun

    @property
    def sharpe_gap(self):
        return self.runs["Kalman"].test["sharpe"] - self.runs["Rolling OLS"].test["sharpe"]


def default_train_end(index, fraction=DEFAULT_TRAIN_FRACTION):
    return index[int(len(index) * fraction) - 1]


def param_grid(model):
    if model == "kalman":
        extra = [{"delta": d} for d in DELTA_GRID]
    else:
        extra = [{"ols_window": w} for w in OLS_WINDOW_GRID]
    return [
        {"entry_threshold": e, "exit_threshold": x, **p}
        for e, x, p in product(ENTRY_GRID, EXIT_GRID, extra)
    ]


def tune(data, train_end, model, base_params):
    """
    Pick the grid point with the highest Sharpe on the train window. Only
    bars up to train_end are passed to the backtester, so nothing after it
    can influence the choice.
    """
    train = data.loc[:train_end]
    warmup = base_params.get("warmup", DEFAULT_WARMUP)
    scored_from = train.index[warmup]
    best, best_sharpe = None, float("-inf")
    for params in param_grid(model):
        result = Backtester(train, model=model, **base_params, **params).run()
        sharpe = compute_metrics(result, start=scored_from)["sharpe"]
        if sharpe > best_sharpe:
            best, best_sharpe = params, sharpe
    return best


def evaluate(data, train_end=None, tuned=True, kalman_params=None, ols_params=None, **base_params):
    """
    Run Kalman and rolling OLS on the full series with parameters chosen on
    the train window (or given), and report metrics for each window. Both
    models trade continuously across the split; the test metrics start
    from the equity at the end of the train window.
    """
    if train_end is None:
        train_end = default_train_end(data.index)
    train_end = pd.Timestamp(train_end)
    train = data.loc[:train_end]
    test_index = data.index[data.index > train_end]
    if len(test_index) < 2:
        raise DataError("The test window needs at least two bars")
    warmup = base_params.get("warmup", DEFAULT_WARMUP)
    if len(train) <= warmup:
        raise DataError(f"The train window needs more than {warmup} bars")

    given = {"kalman": kalman_params or {}, "ols": ols_params or {}}
    train_start = train.index[warmup]
    test_start = test_index[0]
    runs = {}
    for name, model in (("Kalman", "kalman"), ("Rolling OLS", "ols")):
        params = tune(data, train_end, model, base_params) if tuned else given[model]
        result = Backtester(data, model=model, **base_params, **params).run()
        runs[name] = ModelRun(
            params=params,
            result=result,
            train=compute_metrics(result, start=train_start, end=train_end),
            test=compute_metrics(result, start=test_start),
        )
    # After the backtests, which reject degenerate prices with a clear error.
    stats = pair_stats(train["asset_y"], train["asset_x"])
    evaluation = Evaluation(
        warmup_start=data.index[0],
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_index[-1],
        stats=stats,
        runs=runs,
    )
    return evaluation


def takeaway(evaluation):
    gap = evaluation.sharpe_gap
    verb = "beat" if gap > 0 else "trailed"
    return (
        f"Kalman {verb} rolling OLS by {abs(gap):.2f} Sharpe on the test window "
        f"({evaluation.test_start:%Y-%m-%d} to {evaluation.test_end:%Y-%m-%d}), after costs."
    )
