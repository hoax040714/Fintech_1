"""因子IC计算与样本外策略回测。"""

import numpy as np
import pandas as pd

from price_source import RESULT_DIR
from factors import CROSS_SECTION_FEATURES, HORIZONS
from extended_factors import DAILY_FEATURES
from backtest_metrics import maximum_drawdown


def calculate_factor_ic(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """逐日计算股票横截面IC，再按日期汇总。"""
    rows = []

    for horizon in HORIZONS:
        target = f"future_return_{horizon}d"
        mature = features.dropna(subset=[target])

        for date, day in mature.groupby("date"):
            # 宏观/日历/全球共同因子同日没有横截面差异，不能用截面IC评估。
            factor_values = day[CROSS_SECTION_FEATURES]
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


def calculate_time_factor_ic(features: pd.DataFrame) -> pd.DataFrame:
    """共同因子对股票池等权未来收益的时间序列IC；训练/验证/测试分开计算。"""
    common = features.groupby("date", sort=True)[DAILY_FEATURES].first()
    rows = []
    for horizon in HORIZONS:
        target = features.groupby("date")[f"future_return_{horizon}d"].mean()
        sample = common.join(target.rename("target")).dropna(subset=["target"])
        dates = sample.index
        cut1, cut2 = int(len(dates) * .70), int(len(dates) * .85)
        partitions = {"train": dates[:max(0, cut1 - horizon)],
                      "validation": dates[cut1:max(cut1, cut2 - horizon)], "test": dates[cut2:]}
        for partition, selected in partitions.items():
            part = sample.loc[selected]
            for factor in DAILY_FEATURES:
                pair = part[[factor, "target"]].dropna()
                valid = len(pair) >= 20 and pair[factor].nunique() > 1 and pair["target"].nunique() > 1
                rows.append({"horizon": horizon, "factor": factor, "partition": partition,
                             "ic": pair[factor].corr(pair["target"]) if valid else np.nan,
                             "n_dates": len(pair), "ic_scope": "time_series",
                             "target": "股票池当日有成熟标签股票的等权未来收益"})
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULT_DIR / "factor_time_ic_summary.csv", index=False, encoding="utf-8-sig")
    return summary


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

        summary_rows.append(
            {
                "model": model,
                "horizon": horizon,
                "periods": len(returns),
                "total_return": strategy_nav - 1,
                "benchmark_total_return": benchmark_nav - 1,
                "annualized_sharpe": sharpe,
                "max_drawdown": maximum_drawdown(model_nav),
            }
        )

    nav = pd.DataFrame(nav_rows)
    summary = pd.DataFrame(summary_rows)
    nav.to_csv(RESULT_DIR / "backtest_nav.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(RESULT_DIR / "backtest_summary.csv", index=False, encoding="utf-8-sig")
    return nav, summary
