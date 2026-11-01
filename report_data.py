"""网页与HTML共用的已完成实验读取、校验及展示口径。"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import pandas as pd

from experiment_store import experiment_id, sha256_file
from price_source import RESULT_DIR, PRICE_FILE, FEATURE_FILE


MODEL_LABELS = {
    "ols": "OLS近似", "ols_huber": "Huber", "ols3": "三因子回归", "pls": "PLS",
    "lasso": "Lasso", "ridge": "Ridge", "enet": "ElasticNet", "gbrt": "GBRT",
    "random_forest": "Random Forest", "lightgbm": "LightGBM", "xgboost": "XGBoost",
    "lstm": "LSTM", "transformer": "Transformer", "realmlp": "RealMLP", "tabpfn": "TabPFN",
}
STRATEGY_LABELS = {
    "long_short_5v5": "多空组合（多5只、空5只）",
    "long_top5_vs_equal_weight": "前5名相对股票池等权",
}
STRATEGY_NOTES = {
    "long_short_5v5": "每期收益为预测最高5只的等权收益减去最低5只的等权收益，再扣除换手成本。",
    "long_top5_vs_equal_weight": "每期收益为预测最高5只的等权收益减去全股票池等权收益，再扣除换手成本；曲线为相对收益序列的复利值。",
}
COST_NOTE = "每期成本=费用率×换手率；10 bp=0.1%，20 bp=0.2%，并非每次固定扣除该比例。"
IC_NOTE = "这里的IC和Rank IC衡量模型预测值与真实未来收益的截面相关性，不是单个因子的IC。"
PERCENT_COLUMNS = {"mae", "rmse", "validation_mae", "validation_rmse", "test_mae",
                   "direction_accuracy", "predicted_return", "cumulative_return",
                   "max_drawdown", "average_turnover", "net_return", "gross_return", "cost",
                   "positive_ratio", "turnover", "pca_explained_variance"}
COLUMN_LABELS = {
    "model": "模型", "horizon": "预测周期", "code": "股票代码", "name": "股票名称",
    "date": "日期", "close": "参考收盘价", "predicted_return": "预测收益率",
    "reference_target_price": "参考目标价", "best_params": "验证最优参数",
    "mae": "MAE", "rmse": "RMSE", "r2": "R²", "r2_oos_zero": "相对零收益基线R²",
    "direction_accuracy": "方向准确率", "prediction_correlation": "预测相关系数",
    "validation_mae": "验证MAE", "validation_rmse": "验证RMSE", "test_mae": "测试MAE",
    "n_train": "训练样本数", "n_validation": "验证样本数", "n_test": "测试样本数",
    "n_train_stocks": "训练有效股票数", "n_validation_stocks": "验证有效股票数",
    "n_test_stocks": "测试有效股票数", "test_start": "测试开始", "test_end": "测试结束",
    "selected_factor_count": "所选因子数", "model_input_dimensions": "模型输入维数",
    "selection_config": "因子方案", "stage": "拟合阶段", "factor": "因子",
    "pca_explained_variance": "PCA方差保留率",
    "importance": "权重/重要性", "importance_method": "解释口径", "factor_type": "类型",
    "definition": "定义", "metric": "相关口径", "mean": "平均相关系数",
    "std": "波动", "icir": "ICIR", "t_stat": "t统计量", "n_days": "有效日期数",
    "positive_ratio": "正相关占比", "pearson_ic": "Pearson IC", "rank_ic": "Rank IC",
    "n_stocks": "有效股票数", "correlation": "逐股预测相关系数", "strategy": "组合策略",
    "cost_bps": "费用设置", "periods": "调仓期数", "cumulative_return": "累计净收益/相对收益",
    "annualized_sharpe": "年化Sharpe", "max_drawdown": "最大回撤",
    "average_turnover": "平均换手率", "nav": "净值/相对收益复利值",
    "gross_return": "扣费前收益", "net_return": "扣费后收益", "turnover": "换手率",
    "cost": "当期费用", "group": "因子组", "n_factors": "因子数量",
    "latest_prediction": "最新预测", "feature_importance": "因子重要性",
}

TABLE_SCHEMAS = {
    "model_metrics.csv": {"model", "horizon", "mae", "rmse", "r2", "direction_accuracy", "prediction_correlation"},
    "latest_predictions.csv": {"model", "horizon", "date", "code", "name", "close", "predicted_return", "reference_target_price"},
    "feature_importance.csv": {"model", "horizon", "factor", "importance"},
    "preprocessing_summary.csv": {"model", "horizon", "stage", "selected_factor_count", "model_input_dimensions"},
    "ic_daily.csv": {"model", "horizon", "date", "pearson_ic", "rank_ic"},
    "ic_summary.csv": {"model", "horizon", "metric", "mean", "icir", "n_days"},
    "per_stock_metrics.csv": {"model", "horizon", "code", "n_test", "mae", "correlation"},
    "backtest_nav.csv": {"model", "horizon", "strategy", "cost_bps", "date", "nav"},
    "backtest_summary.csv": {"model", "horizon", "strategy", "cost_bps", "periods", "cumulative_return", "annualized_sharpe", "max_drawdown", "average_turnover"},
    "factor_group_ablation.csv": {"horizon", "group", "n_factors", "validation_mae", "test_mae"},
}


class ReportError(ValueError):
    """尚无当前实验，或输入/结果无法通过一致性校验。"""


def model_label(name):
    return MODEL_LABELS.get(str(name), str(name))


def strategy_label(name):
    return STRATEGY_LABELS.get(str(name), str(name))


def nav_label(strategy):
    return "相对收益复利曲线" if strategy == "long_top5_vs_equal_weight" else "策略净值"


def existing_columns(frame, names):
    return [name for name in names if name in frame.columns]


def _metadata(path):
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8-sig"))
    except (OSError, ValueError, UnicodeError) as error:
        raise ReportError("尚无可读取的当前实验结果，请先完成一次训练。") from error
    if not isinstance(value, dict):
        raise ReportError("实验说明文件格式不正确。")
    return raw, value


def _artifact_path(folder, name):
    if not isinstance(name, str) or Path(name).name != name or name in {"", ".", ".."}:
        raise ReportError("实验产物清单包含无效文件名。")
    target = (folder / name).resolve()
    if target.parent != folder.resolve():
        raise ReportError("实验产物指向了结果目录之外。")
    return target


def _stat(path):
    try:
        item = path.stat()
        return item.st_size, item.st_mtime_ns
    except OSError:
        return None


def report_cache_key(result_dir=RESULT_DIR, price_file=PRICE_FILE, feature_file=FEATURE_FILE):
    """结果发布或文件变化时自动失效，不仅以CSV文件名作为缓存键。"""
    folder = Path(result_dir).resolve()
    raw, metadata = _metadata(folder / "metadata.json")
    manifest = metadata.get("artifacts")
    artifacts = manifest.get("results", {}) if isinstance(manifest, dict) else {}
    if not isinstance(artifacts, dict):
        artifacts = {}
    return (str(folder), hashlib.sha256(raw).hexdigest(),
            tuple((str(path), _stat(path)) for path in (Path(price_file), Path(feature_file))),
            tuple((name, _stat(_artifact_path(folder, name))) for name in sorted(artifacts)))


@dataclass
class ExperimentReport:
    metadata: dict
    tables: dict
    warnings: list

    def table(self, name):
        return self.tables.get(name, pd.DataFrame()).copy()

    @property
    def experiment_id(self):
        return self.metadata["experiment_id"]

    @property
    def stock_count(self):
        return len(self.metadata["experiment_identity"]["stock_codes"])

    @property
    def feature_count(self):
        return len(self.metadata["experiment_identity"].get("features", ()))

    @property
    def models(self):
        return sorted(self.table("model_metrics.csv").model.unique().tolist(), key=model_label)


def load_report(result_dir=RESULT_DIR, price_file=PRICE_FILE, feature_file=FEATURE_FILE):
    """只读产物清单中的当前实验文件，不回退到历史IC或旧版CSV。"""
    folder = Path(result_dir).resolve()
    raw, metadata = _metadata(folder / "metadata.json")
    identity = metadata.get("experiment_identity")
    if metadata.get("status") != "complete" or not isinstance(identity, dict):
        raise ReportError("当前目录仍是旧版结果或尚未成功发布新实验；完成新版训练后即可展示，历史文件不会混入。")
    try:
        if experiment_id(identity) != metadata.get("experiment_id"):
            raise ReportError("实验标识与记录不一致，请检查结果文件。")
        stocks = identity["stock_codes"]
        if metadata.get("stock_count") != len(stocks) or metadata.get("stock_codes") != stocks:
            raise ReportError("实验股票池信息不一致，拒绝展示混合结果。")
        for kind, path in (("prices", Path(price_file)), ("features", Path(feature_file))):
            if not path.is_file() or sha256_file(path) != identity["input_sha256"][kind]:
                raise ReportError("当前行情或因子文件与实验不一致，请使用对应的数据快照。")
        artifacts = metadata["artifacts"]["results"]
        if not isinstance(artifacts, dict):
            raise ReportError("实验缺少有效产物清单。")
        horizons = set(metadata["horizons"])
        tables, warnings = {}, []
        for name, required in TABLE_SCHEMAS.items():
            if name not in artifacts:
                continue
            path = _artifact_path(folder, name)
            if not path.is_file() or sha256_file(path) != artifacts[name]:
                raise ReportError(f"实验产物缺失或已被修改：{name}。")
            try:
                frame = pd.read_csv(path, dtype={"code": str})
            except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeError) as error:
                raise ReportError(f"实验结果表无法读取：{name}。") from error
            if not frame.empty:
                missing = required - set(frame.columns)
                if missing:
                    raise ReportError(f"{name} 缺少当前流程字段：{', '.join(sorted(missing))}。")
                if "experiment_id" not in frame or not frame.experiment_id.eq(metadata["experiment_id"]).all():
                    raise ReportError(f"{name} 包含不同实验的记录。")
                if not frame.horizon.isin(horizons).all():
                    raise ReportError(f"{name} 包含实验记录之外的预测周期。")
                if "date" in frame:
                    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
                    if frame.date.isna().any():
                        raise ReportError(f"{name} 包含缺失日期。")
                if "code" in frame:
                    frame["code"] = frame.code.str.zfill(6)
            tables[name] = frame
        if "model_metrics.csv" not in tables or tables["model_metrics.csv"].empty:
            raise ReportError("当前实验还没有有效的模型评价结果。")
        if metadata.get("neural_enabled"):
            warnings.append("LSTM、Transformer目前仅提供测试评价；没有生产模型最新预测或内置因子重要性时会明确标注。")
        if "factor_group_ablation.csv" not in tables:
            warnings.append("本次实验未生成因子组消融结果；当前流程只在1日或5日实验中生成此项。")
        if (folder / "metadata.json").read_bytes() != raw:
            raise ReportError("结果刚刚更新，请刷新页面后重试。")
        return ExperimentReport(metadata, tables, warnings)
    except ReportError:
        raise
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ReportError(f"实验结果校验失败：{error}") from error


def model_capabilities(report):
    metrics = report.table("model_metrics.csv")
    latest = report.table("latest_predictions.csv")
    importance = report.table("feature_importance.csv")
    rows = []
    for item in metrics[["model", "horizon"]].drop_duplicates().itertuples(index=False):
        predicted = not latest.empty and ((latest.model == item.model) & (latest.horizon == item.horizon)).any()
        available = importance[(importance.model == item.model) & (importance.horizon == item.horizon)] if not importance.empty else pd.DataFrame()
        if "importance_method" in available:
            available = available[available.importance_method != "not_available"]
        rows.append({"model": item.model, "horizon": item.horizon,
                     "latest_prediction": "已生成" if predicted else "未生成（仅有测试评价）",
                     "feature_importance": "已提供" if not available.empty else "未提供"})
    return pd.DataFrame(rows)


def select_backtest(report, model, horizon, strategy, cost_bps):
    def select(name):
        frame = report.table(name)
        if frame.empty:
            return frame
        return frame[(frame.model == model) & (frame.horizon == horizon)
                     & (frame.strategy == strategy) & (frame.cost_bps == cost_bps)].copy()
    return select("backtest_nav.csv").sort_values("date") if not report.table("backtest_nav.csv").empty else pd.DataFrame(), select("backtest_summary.csv")


def importance_view(report, model, horizon, top_n=20):
    frame = report.table("feature_importance.csv")
    if frame.empty:
        return frame
    frame = frame[(frame.model == model) & (frame.horizon == horizon)].copy()
    if "importance_method" in frame:
        frame = frame[frame.importance_method != "not_available"]
    frame["abs_importance"] = frame.importance.abs()
    return frame.nlargest(top_n, "abs_importance")
