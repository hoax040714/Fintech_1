"""项目主程序：python main.py"""

import argparse

from price_source import load_prices, PRICE_FILE
from factors import make_features


def run_project(n_stocks: int | None = None, end_date: str | None = None,
                features_only: bool = False, selection_only: bool = False,
                mae_tolerance: float = 0., offline: bool = False,
                refresh: bool = False, data_only: bool = False) -> None:
    if not 0 <= mae_tolerance <= .1:
        raise ValueError("--mae-tolerance 必须位于0至0.1；0表示严格优先验证MAE最小")
    if n_stocks is not None and not 1 <= n_stocks <= 300:
        raise ValueError("--stocks 必须为1至300")
    if offline and (refresh or data_only):
        raise ValueError("--offline 不能与 --refresh 或 --data-only 同时使用")
    if not offline:
        # 下载与建模之间以Parquet为唯一接口，不能绕过落盘直接使用内存中的合并表。
        from data import download_dataset
        download_dataset(n_stocks=n_stocks or 300, end_date=end_date, refresh=refresh)
    if data_only:
        print(f"数据准备完成：{PRICE_FILE}；未构建因子、未训练模型。", flush=True)
        return
    prices = load_prices()
    if n_stocks is not None:
        if n_stocks < 1:
            raise ValueError("--stocks 必须为正整数")
        codes = sorted(prices["code"].unique())[:n_stocks]
        prices = prices[prices["code"].isin(codes)].copy()
    if end_date is not None:
        import pandas as pd
        prices = prices[prices["date"] <= pd.Timestamp(end_date)].copy()
    if prices.empty:
        raise ValueError("筛选后没有行情数据")
    print(f"唯一数据源：{PRICE_FILE}；{prices['code'].nunique()}只股票，{len(prices):,}行", flush=True)
    features = make_features(prices)
    if features_only:
        print("因子、字段用途和质量检查文件已生成；未训练模型。", flush=True)
        return
    # 仅生成因子时无需导入模型依赖，也不会改写既有训练结果。
    from models import train_models, audit_factor_selection
    if selection_only:
        audit_factor_selection(features)
        print("训练期因子筛选与PCA维数诊断已生成；未选择验证赢家，未训练最终预测模型。", flush=True)
        return
    from evaluate import calculate_factor_ic, calculate_time_factor_ic, run_backtest
    metrics, predictions = train_models(features, relative_tolerance=mae_tolerance)
    calculate_factor_ic(features)
    calculate_time_factor_ic(features)
    run_backtest(predictions)

    print("\n完成。模型指标：")
    columns = ["model", "horizon", "mae", "r2", "direction_accuracy"]
    print(metrics[columns].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="在线下载生成prices.parquet，再构建因子、筛选降维与预测")
    parser.add_argument("--stocks", type=int, default=None, help="在线默认下载沪深300全部股票；可指定前N只（按代码），离线筛选已有股票")
    parser.add_argument("--end-date", default=None, help="下载截止日；离线模式筛选已有日期，格式YYYYMMDD")
    parser.add_argument("--offline", action="store_true", help="跳过下载，复用已有prices.parquet")
    parser.add_argument("--refresh", action="store_true", help="忽略原始下载缓存，重新下载全部来源")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--data-only", action="store_true", help="只在线下载并生成prices.parquet")
    mode.add_argument("--features-only", action="store_true", help="只构建因子并输出字段用途/质量报告")
    mode.add_argument("--selection-only", action="store_true", help="只构建因子并审计训练期筛选/PCA候选维数，不选择验证赢家")
    parser.add_argument("--mae-tolerance", type=float, default=0.,
                        help="相对最优验证MAE允许的误差上浮，范围0至0.1；默认0，仅同精度时优先更低维")
    args = parser.parse_args()
    run_project(args.stocks, args.end_date, args.features_only, args.selection_only, args.mae_tolerance,
                args.offline, args.refresh, args.data_only)


if __name__ == "__main__":
    main()

