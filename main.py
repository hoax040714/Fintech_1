"""项目主程序：python main.py"""

import argparse

from data import add_market_context, download_prices, get_end_date, get_stock_pool
from evaluate import calculate_factor_ic, run_backtest
from factors import HORIZONS, make_features
from models import DEFAULT_MODELS, MODEL_LABELS, train_models


def run_project(n_stocks: int = 300, refresh: bool = False, end_date: str | None = None,
                data_only: bool = False, model_names=None, horizons=None,
                nn_epochs=8, device="cpu") -> None:
    end_date = get_end_date(end_date)
    pool = get_stock_pool(n_stocks=n_stocks, refresh=refresh)
    prices = download_prices(pool, end_date=end_date, refresh=refresh)
    prices = add_market_context(
        prices,
        pool,
        end_date=end_date,
        refresh=refresh,
    )
    if data_only:
        print(f"数据匹配完成：{prices['code'].nunique()}只股票，{len(prices):,}行，"
              f"{prices['date'].min().date()}至{prices['date'].max().date()}")
        return
    features = make_features(prices)
    metrics, predictions = train_models(features, model_names, horizons, nn_epochs, device)
    calculate_factor_ic(features)
    run_backtest(predictions)

    print("\n完成。模型指标：")
    columns = ["model", "horizon", "mae", "r2", "direction_accuracy"]
    print(metrics[columns].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="沪深300成分股收益预测（2020年至今）")
    parser.add_argument("--stocks", type=int, default=300, help="默认全部300只；可指定较小数量检查流程")
    parser.add_argument("--end-date", default=None, help="下载截止日，格式YYYYMMDD")
    parser.add_argument("--refresh", action="store_true", help="重新下载股票池和行情")
    parser.add_argument("--data-only", action="store_true", help="只下载并匹配数据，不训练模型")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_LABELS), help="只运行指定模型，名称用空格分隔")
    parser.add_argument("--horizons", nargs="+", type=int, choices=HORIZONS, help="只训练指定预测周期")
    parser.add_argument("--neural", action="store_true", help="增加 LSTM 和 Transformer")
    parser.add_argument("--modern", action="store_true", help="增加 RealMLP；TabPFN须通过 --models 显式指定")
    parser.add_argument("--nn-epochs", type=int, default=8, help="序列网络最多训练轮数，默认8")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="4个可选模型的运行设备")
    args = parser.parse_args()
    names = list(args.models or DEFAULT_MODELS)
    if args.neural:
        names.extend(["lstm", "transformer"])
    if args.modern:
        names.append("realmlp")
    names = list(dict.fromkeys(names))
    run_project(args.stocks, args.refresh, args.end_date, args.data_only,
                names, args.horizons, args.nn_epochs, args.device)


if __name__ == "__main__":
    main()

