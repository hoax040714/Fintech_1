"""因子IC计算与样本外策略回测。"""

import numpy as np
import pandas as pd

from data import RESULT_DIR
from factors import FEATURES, HORIZONS


def calculate_factor_ic(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """逐日计算股票横截面IC，再按日期汇总。"""
    rows = []

    for horizon in HORIZONS:
        target = f"future_return_{horizon}d"
        mature = features.dropna(subset=[target])

        for date, day in mature.groupby("date"):
            factor_values = day[FEATURES]
            counts = factor_values.notna().sum()
            correlations = factor_values.corrwith(day[target])
            valid = correlations[(counts >= 10) & correlations.notna()]

            rows.extend(
                (date, horizon, factor, ic, int(counts[factor]))
                for factor, ic in valid.items()
            )

    daily = pd.DataFrame(rows, columns=["date", "horizon", "factor", "ic", "n_stocks"])
    summary = (
        daily.groupby(["horizon", "factor"])["ic"]
        .agg(
            mean_ic="mean",
            ic_std="std",
            positive_ratio=lambda values: (values > 0).mean(),
            n_dates="count",
        )
        .reset_index()
    )
    summary["abs_mean_ic"] = summary["mean_ic"].abs()
    summary = summary.sort_values(["horizon", "abs_mean_ic"], ascending=[True, False])

    daily.to_csv(RESULT_DIR / "factor_ic_daily.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(RESULT_DIR / "factor_ic_summary.csv", index=False, encoding="utf-8-sig")
    return daily, summary


def run_backtest(
    predictions: pd.DataFrame,
    top_n: int = 10,
    cost: float = 0.001,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """每隔预测周期调仓，等权持有预测最高的 top_n 只股票。"""
    nav_rows = []
    summary_rows = []

    for (model, horizon), group in predictions.groupby(["model", "horizon"]):
        dates = sorted(group["date"].unique())
        rebalance_dates = dates[:: int(horizon)]
        strategy_nav = 1.0
        benchmark_nav = 1.0
        period_returns = []

        for date in rebalance_dates:
            day = group[group["date"] == date].dropna(
                subset=["actual_return", "predicted_return"]
            )
            if len(day) < max(top_n, 10):
                continue

            selected = day.nlargest(top_n, "predicted_return")
            strategy_return = float(selected["actual_return"].mean()) - cost
            benchmark_return = float(day["actual_return"].mean()) - cost
            strategy_nav *= 1 + strategy_return
            benchmark_nav *= 1 + benchmark_return
            period_returns.append(strategy_return)

            nav_rows.append(
                {
                    "date": date,
                    "model": model,
                    "horizon": horizon,
                    "strategy_return": strategy_return,
                    "benchmark_return": benchmark_return,
                    "strategy_nav": strategy_nav,
                    "benchmark_nav": benchmark_nav,
                    "selected_count": len(selected),
                }
            )

        returns = np.asarray(period_returns)
        periods_per_year = 252 / horizon
        return_std = returns.std(ddof=1)
        sharpe = np.sqrt(periods_per_year) * returns.mean() / return_std

        model_nav = [
            row["strategy_nav"]
            for row in nav_rows
            if row["model"] == model and row["horizon"] == horizon
        ]
        model_nav = np.asarray(model_nav)
        drawdown = model_nav / np.maximum.accumulate(model_nav) - 1

        summary_rows.append(
            {
                "model": model,
                "horizon": horizon,
                "periods": len(returns),
                "total_return": strategy_nav - 1,
                "benchmark_total_return": benchmark_nav - 1,
                "annualized_sharpe": sharpe,
                "max_drawdown": float(drawdown.min()),
            }
        )

    nav = pd.DataFrame(nav_rows)
    summary = pd.DataFrame(summary_rows)
    nav.to_csv(RESULT_DIR / "backtest_nav.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(RESULT_DIR / "backtest_summary.csv", index=False, encoding="utf-8-sig")
    return nav, summary
