"""使用本地行情对应的全部股票运行417因子实验。示例：python main.py --horizons 1。"""

import argparse
from time import perf_counter

import pandas as pd

from factors import FEATURES, HORIZONS
from price_source import FEATURE_FILE, PRICE_FILE
from run_progress import duration_text, log_progress, progress_stage

# 时间范围：包含起止日期
START_DATE = "2025-01-01"
END_DATE = "2025-12-31"

# 按代码排序选择前多少只；None 表示不限制数量
STOCK_COUNT = 10

# 指定股票名单时优先使用名单；None 表示按上面的数量选择
STOCK_CODES = None

# 例如指定股票时改成：
# STOCK_CODES = ["000001", "000002", "000063", ...]


def load_feature_panel() -> pd.DataFrame:
    """读取全部因子记录，并确认覆盖行情表的完整股票与日期主键。"""
    if not FEATURE_FILE.is_file():
        raise FileNotFoundError(f"找不到预先生成的因子文件：{FEATURE_FILE}")
    with progress_stage("[数据读取] 读取本地因子文件") as stage:
        frame = pd.read_parquet(FEATURE_FILE)
        stage.detail = f"{len(frame):,}行、{len(frame.columns)}列"
    required = ["date", "code", *FEATURES, *[f"future_return_{h}d" for h in HORIZONS]]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"因子文件缺少{len(missing)}个必要字段，示例：{missing[:8]}")
    if not PRICE_FILE.is_file():
        raise FileNotFoundError(f"找不到用于核对完整股票池的行情文件：{PRICE_FILE}")
    with progress_stage("[数据读取] 读取行情日期和股票代码"):
        source = pd.read_parquet(PRICE_FILE, columns=["date", "code"])
    log_progress("[数据校验] 检查日期、股票代码、重复记录及两张表的覆盖范围")
    for panel in (frame, source):
        if panel.empty or panel[["date", "code"]].isna().any().any():
            raise ValueError("行情表和因子表必须非空，且股票代码、日期不能缺失")
        panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
        panel["code"] = panel["code"].astype(str).str.zfill(6)
        if not panel["code"].str.fullmatch(r"\d{6}").all():
            raise ValueError("股票代码必须为六位数字")
        if panel.duplicated(["date", "code"]).any():
            raise ValueError("行情表或因子表存在重复的日期、股票代码主键")
    source_keys = pd.MultiIndex.from_frame(source[["date", "code"]])
    feature_keys = pd.MultiIndex.from_frame(frame[["date", "code"]])
    missing_keys = source_keys.difference(feature_keys)
    extra_keys = feature_keys.difference(source_keys)
    if len(missing_keys) or len(extra_keys):
        raise ValueError(
            f"因子表与行情表覆盖范围不一致：缺少{len(missing_keys)}条、多出{len(extra_keys)}条。"
            "请先基于完整prices.parquet重新生成因子表，不能只训练部分股票或日期。"
        )
    log_progress(f"[数据校验] 通过，源数据共{source.code.nunique()}只股票、{len(source):,}行")
    log_progress(f"[范围筛选] 日期 {START_DATE} 至 {END_DATE}；"
                 f"股票设置：{STOCK_CODES if STOCK_CODES is not None else STOCK_COUNT or '全部'}")
    # 完成原有完整性校验后，再筛选实验范围
    frame = frame.loc[
        frame["date"].between(
            pd.Timestamp(START_DATE),
            pd.Timestamp(END_DATE),
        )
    ].copy()

    if STOCK_CODES is not None:
        codes = [str(code).zfill(6) for code in STOCK_CODES]
        missing = set(codes) - set(frame["code"])
        if missing:
            raise ValueError(f"这些股票在指定时间内没有数据：{sorted(missing)}")
    elif STOCK_COUNT is not None:
        if STOCK_COUNT <= 0:
            raise ValueError("STOCK_COUNT 必须大于0，或设置为None")
        codes = sorted(frame["code"].unique())[:STOCK_COUNT]
    else:
        codes = None

    if codes is not None:
        frame = frame.loc[frame["code"].isin(codes)].copy()

    if frame.empty:
        raise ValueError("指定的时间范围和股票范围没有可用数据")

    frame.sort_values(["date", "code"], inplace=True, ignore_index=True)
    log_progress(f"[范围筛选] 实际使用{frame.code.nunique()}只股票、{len(frame):,}行、"
                 f"{frame.date.nunique()}个日期；"
                 f"{frame.date.min():%Y-%m-%d} 至 {frame.date.max():%Y-%m-%d}")
    return frame


def run_project(horizons: tuple[int, ...], selection_only: bool = False,
                neural: bool = False, modern: bool = False, model_device: str = "cpu",
                nn_seeds: int = 3) -> None:
    """让行情表中的全部股票进入时间切分和训练流程，不重算或覆盖因子数据。"""
    started = perf_counter()
    log_progress(f"[运行开始] 预测周期：{', '.join(str(h) + '日' for h in horizons)}；"
                 f"神经网络：{'开启' if neural else '关闭'}；现代表格模型：{'开启' if modern else '关闭'}")
    import factors
    factors.HORIZONS = horizons
    with progress_stage("[程序准备] 加载训练与评价模块"):
        from models import train_models, audit_factor_selection

    features = load_feature_panel()
    print(f"读取 {FEATURE_FILE.name}: {len(features):,} 行，{features.code.nunique()} 只股票，"
          f"{len(FEATURES)} 个候选因子", flush=True)
    if selection_only:
        audit_factor_selection(features)
        log_progress(f"[运行完成] 因子筛选审计完成，总耗时{duration_text(perf_counter() - started)}")
        return
    metrics, _ = train_models(features, neural=neural, modern=modern,
                              model_device=model_device, nn_seeds=nn_seeds)
    print("\n测试集指标：")
    cols = [c for c in ("model", "horizon", "best_params", "mae", "r2_oos_zero", "direction_accuracy")
            if c in metrics]
    print(metrics[cols].to_string(index=False))
    log_progress(f"[运行完成] 实验结果已成功发布，总耗时{duration_text(perf_counter() - started)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="用417因子Parquet中的全部股票运行收益预测模型")
    parser.add_argument("--horizons", default="1,5,30,60", help="逗号分隔预测周期，默认1,5,30,60日")
    parser.add_argument("--selection-only", action="store_true", help="仅生成训练期筛选审计，不训练模型")
    parser.add_argument("--neural", action="store_true", help="运行LSTM和Transformer")
    parser.add_argument("--modern", action="store_true", help="运行RealMLP和TabPFN")
    parser.add_argument("--model-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--nn-seeds", type=int, default=3)
    args = parser.parse_args()
    horizons = tuple(int(x.strip()) for x in args.horizons.split(",") if x.strip())
    if not horizons or any(h not in HORIZONS for h in horizons):
        parser.error("请提供至少一个有效周期（1,5,30,60）")
    run_project(horizons, args.selection_only, args.neural, args.modern,
                args.model_device, args.nn_seeds)


if __name__ == "__main__":
    main()
