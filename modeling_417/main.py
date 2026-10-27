# 修改记录：增加离线使用417因子Parquet、按代码筛选股票和按周期运行的入口。
"""独立模型实验入口。示例：python -m modeling_417.main --horizons 1"""

import argparse
import json
from pathlib import Path

import pandas as pd

from factors import FEATURES, HORIZONS
from price_source import FEATURE_FILE, RESULT_DIR


def load_feature_panel(codes: list[str] | None = None) -> pd.DataFrame:
    """读取全股票池事先生成的因子，再筛选建模股票，保留行业排名计算口径。"""
    if not FEATURE_FILE.is_file():
        raise FileNotFoundError(f"找不到预先生成的因子文件：{FEATURE_FILE}")
    frame = pd.read_parquet(FEATURE_FILE)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    missing = [name for name in FEATURES if name not in frame.columns]
    if missing:
        raise ValueError(f"因子文件缺少{len(missing)}列，示例：{missing[:8]}")
    if codes:
        chosen = [str(code).zfill(6) for code in codes]
        present = set(frame["code"].unique())
        absent = sorted(set(chosen) - present)
        if absent:
            raise ValueError(f"因子文件找不到股票代码：{absent}")
        frame = frame[frame["code"].isin(chosen)].copy()
    return frame.sort_values(["date", "code"]).reset_index(drop=True)


def run_project(codes: list[str], horizons: tuple[int, ...], selection_only: bool = False,
                neural: bool = False, modern: bool = False, model_device: str = "cpu",
                nn_seeds: int = 3) -> None:
    """使用同学已生成的完整因子表训练所选股票，不重算或覆盖因子数据。"""
    import factors
    factors.HORIZONS = horizons
    from .models import train_models, audit_factor_selection

    features = load_feature_panel(codes)
    print(f"读取 {FEATURE_FILE.name}: {len(features):,} 行，{features.code.nunique()} 只股票，"
          f"{len(FEATURES)} 个候选因子", flush=True)
    if selection_only:
        audit_factor_selection(features)
        print("因子筛选审计完成。", flush=True)
        return
    metrics, _ = train_models(features, neural=neural, modern=modern,
                              model_device=model_device, nn_seeds=nn_seeds)
    print("\n测试集指标：")
    cols = [c for c in ("model", "horizon", "best_params", "mae", "r2_oos_zero", "direction_accuracy")
            if c in metrics]
    print(metrics[cols].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="用417因子Parquet运行股票收益预测模型")
    parser.add_argument("--codes", default="300750,300308,600519,601318,300502,601899,600036,000333,603259,688256",
                        help="逗号分隔的股票代码；默认任务书的沪深300权重前十")
    parser.add_argument("--horizons", default="1,5,30,60", help="逗号分隔预测周期，默认1,5,30,60日")
    parser.add_argument("--selection-only", action="store_true", help="仅生成训练期筛选审计，不训练模型")
    parser.add_argument("--neural", action="store_true", help="运行LSTM和Transformer")
    parser.add_argument("--modern", action="store_true", help="运行RealMLP和TabPFN")
    parser.add_argument("--model-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--nn-seeds", type=int, default=3)
    args = parser.parse_args()
    codes = [x.strip() for x in args.codes.split(",") if x.strip()]
    horizons = tuple(int(x.strip()) for x in args.horizons.split(",") if x.strip())
    if not codes or not horizons or any(h not in HORIZONS for h in horizons):
        parser.error("请提供至少一个代码和有效周期（1,5,30,60）")
    run_project(codes, horizons, args.selection_only, args.neural, args.modern,
                args.model_device, args.nn_seeds)


if __name__ == "__main__":
    main()
