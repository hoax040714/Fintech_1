"""训练期因子筛选、相关性去冗余和可序列化的PCA预处理。

监督评分只使用调用者传入的训练标签；公共因子按日期评分，个股因子按日截面评分。
候选因子数量/PCA方案由 models.py 在验证集比较，测试标签不参与选择。
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from extended_factors import DAILY_FEATURES


PREPROCESSING_VERSION = "train-factor-screen-pca-v1"


@dataclass(frozen=True)
class SelectionConfig:
    name: str
    max_factors: int | None
    pca_variance: float | None = None
    remove_redundancy: bool = True
    fixed_features: tuple[str, ...] | None = None


class FactorScreen:
    """训练期质量/预测能力/稳定性/冗余审计，不读取验证集或测试集。"""

    def __init__(self, missing_limit=.40, dominance_limit=.999, correlation_limit=.95,
                 max_stocks_per_date=80, max_sample_rows=100000, min_dates=60,
                 clip_quantiles=(.005, .995), random_state=42, min_cross_section=10):
        self.missing_limit = missing_limit
        self.dominance_limit = dominance_limit
        self.correlation_limit = correlation_limit
        self.max_stocks_per_date = max_stocks_per_date
        self.max_sample_rows = max_sample_rows
        self.min_dates = min_dates
        self.clip_quantiles = clip_quantiles
        self.random_state = random_state
        # IC有效样本的最低门槛，不是训练股票数量上限。
        self.min_cross_section = min_cross_section

    def _sample_positions(self, dates):
        groups = pd.Series(np.arange(len(dates))).groupby(dates.to_numpy(), sort=True)
        per_date = min(self.max_stocks_per_date, max(1, self.max_sample_rows // groups.ngroups))
        rng = np.random.default_rng(self.random_state)
        positions = []
        for _, group in groups:
            values = group.to_numpy()
            if len(values) > per_date:
                values = np.sort(rng.choice(values, per_date, replace=False))
            positions.extend(values)
        return np.asarray(positions, dtype=int)

    def fit(self, X: pd.DataFrame, y: pd.Series, dates: pd.Series):
        if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
            raise ValueError("因子筛选需要列名唯一的DataFrame")
        if not X.index.equals(y.index) or not X.index.equals(dates.index):
            raise ValueError("因子、标签和日期索引必须完全一致")
        dates = pd.to_datetime(dates, errors="raise")
        if dates.isna().any() or not np.isfinite(y.to_numpy(dtype=float)).all():
            raise ValueError("训练日期或标签存在缺失/无穷值")
        if dates.nunique() < self.min_dates or y.nunique() < 2:
            raise ValueError(f"训练期至少需要{self.min_dates}个日期且标签不能为常量")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.fit_start_ = dates.min()
        self.fit_end_ = dates.max()
        self.n_fit_rows_ = len(X)
        rows = []
        eligible = []
        for name in X:
            values = pd.to_numeric(X[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
            count = values.notna().sum()
            unique = values.nunique()
            missing = 1 - count / len(values)
            dominant = values.value_counts(normalize=True).iloc[0] if count else 1.
            reason = ("all_missing" if count == 0 else "constant" if unique < 2 else
                      "high_missing" if missing > self.missing_limit else
                      "near_constant" if dominant > self.dominance_limit else "eligible")
            rows.append({"factor": name, "missing_pct": missing * 100, "n_unique": unique,
                         "dominant_fraction": dominant, "quality_reason": reason})
            if reason == "eligible":
                eligible.append(name)
        if not eligible:
            raise ValueError("训练期质量过滤后没有可用因子")
        self.sample_positions_ = self._sample_positions(dates)
        sampled = X.iloc[self.sample_positions_][eligible].astype("float32").replace([np.inf, -np.inf], np.nan)
        sample_dates = dates.iloc[self.sample_positions_]
        sample_y = y.iloc[self.sample_positions_]
        quantiles = sampled.quantile(list(self.clip_quantiles))
        unique_counts = pd.Series({row["factor"]: row["n_unique"] for row in rows})
        # 二元事件和低基数日历变量不能被分位缩尾抹掉。
        continuous = unique_counts.reindex(eligible).gt(20)
        self.lower_bounds_ = quantiles.iloc[0].where(continuous, -np.inf)
        self.upper_bounds_ = quantiles.iloc[1].where(continuous, np.inf)
        sampled = sampled.clip(self.lower_bounds_, self.upper_bounds_, axis=1)
        varying = sampled.nunique().gt(1)
        post_constant = varying[~varying].index.tolist()
        eligible = [name for name in eligible if varying[name]]
        if not eligible:
            raise ValueError("训练采样与缩尾后没有变化的因子")
        sampled = sampled[eligible]
        within_date_unique = sampled.groupby(sample_dates).nunique(dropna=True)
        common = set(DAILY_FEATURES).intersection(eligible)
        common.update(within_date_unique.columns[within_date_unique.le(1).all()])
        common = [name for name in eligible if name in common]
        stock = [name for name in eligible if name not in common]
        all_dates = pd.Index(sorted(dates.unique()))
        blocks = [pd.Index(part) for part in np.array_split(all_dates.to_numpy(), 3) if len(part)]
        daily_ics = []
        if stock:
            for date, positions in sampled.groupby(sample_dates, sort=True).indices.items():
                day = sampled.iloc[positions][stock]
                target = sample_y.iloc[positions]
                counts = day.notna().sum()
                std = day.std(ddof=0)
                threshold = day.abs().mean().clip(lower=1) * 1e-10
                valid = (counts >= self.min_cross_section) & (std > threshold)
                corr = day.loc[:, valid].corrwith(target) if target.nunique() > 1 else pd.Series(dtype=float)
                corr.name = date
                daily_ics.append(corr)
        stock_ic = pd.DataFrame(daily_ics).reindex(index=all_dates, columns=stock)
        daily = sampled[common].groupby(sample_dates, sort=True).first() if common else pd.DataFrame(index=all_dates)
        daily_target = y.groupby(dates, sort=True).mean().reindex(all_dates)
        metrics = []
        for name in eligible:
            if name in common:
                values = daily[name].reindex(all_dates)
                pair = pd.concat([values.rename("x"), daily_target.rename("y")], axis=1).dropna()
                overall = pair.x.corr(pair.y) if len(pair) >= 20 and pair.x.nunique() > 1 else np.nan
                block_ics = []
                for block in blocks:
                    part = pair.loc[pair.index.intersection(block)]
                    block_ics.append(part.x.corr(part.y) if len(part) >= 20 and part.x.nunique() > 1 else np.nan)
                valid_dates = len(pair)
                scope = "time_series"
            else:
                values = stock_ic[name]
                overall = values.mean()
                block_ics = [values.reindex(block).mean() for block in blocks]
                valid_dates = values.notna().sum()
                scope = "cross_section"
            valid_blocks = np.asarray([v for v in block_ics if np.isfinite(v)])
            agreement = float(np.mean(np.sign(valid_blocks) == np.sign(overall))) if len(valid_blocks) and np.isfinite(overall) else 0.
            dispersion = float(valid_blocks.std()) if len(valid_blocks) else np.nan
            score = (abs(overall) * agreement * np.sqrt(valid_dates / len(all_dates)) / (1 + dispersion)
                     if np.isfinite(overall) and len(valid_blocks) >= 2 else 0.)
            metrics.append({"factor": name, "ic_scope": scope, "train_ic": overall,
                            "block_1_ic": block_ics[0], "block_2_ic": block_ics[1], "block_3_ic": block_ics[2],
                            "sign_agreement": agreement, "score": score, "valid_dates": int(valid_dates)})
        ranked = pd.DataFrame(metrics)
        # 截面IC与时间序列IC的目标/噪声不同，先在各自口径内转为百分位再合并排序。
        ranked["scope_rank_score"] = ranked.groupby("ic_scope")["score"].rank(pct=True)
        ranked.loc[ranked.score.le(0), "scope_rank_score"] = 0.
        ranked = ranked.sort_values(["scope_rank_score", "score", "factor"], ascending=[False, False, True], kind="stable")
        self.ranked_features_ = ranked.factor.tolist()
        # 相关性在训练样本中位数填充后计算，与模型预处理一致；矩阵乘法避免面板两两循环。
        ranked_values = sampled[self.ranked_features_]
        matrix = ranked_values.fillna(ranked_values.median()).to_numpy(dtype=np.float32, copy=True)
        matrix -= matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
        norms = np.sqrt(np.einsum("ij,ij->j", matrix, matrix, dtype=np.float64))
        matrix /= np.where(norms > 0, norms, 1).astype(np.float32)
        correlations = pd.DataFrame(np.clip(np.abs(matrix.T @ matrix), 0, 1), index=self.ranked_features_, columns=self.ranked_features_)
        independent, redundancy = [], []
        for name in self.ranked_features_:
            if independent:
                corr = correlations.loc[name, independent]
                other = corr.idxmax()
                maximum = float(corr.loc[other])
            else:
                other, maximum = "", 0.
            rejected = maximum >= self.correlation_limit
            redundancy.append({"factor": name, "redundant_with": other if rejected else "",
                               "max_kept_correlation": maximum, "redundant": rejected})
            if not rejected:
                independent.append(name)
        self.independent_features_ = independent
        audit = pd.DataFrame(rows).merge(ranked, on="factor", how="left", validate="one_to_one")
        audit.loc[audit.factor.isin(post_constant), "quality_reason"] = "constant_after_clipping_sample"
        audit = audit.merge(pd.DataFrame(redundancy), on="factor", how="left", validate="one_to_one")
        audit["rank"] = audit.factor.map({name: i + 1 for i, name in enumerate(self.ranked_features_)})
        audit["clip_lower"] = audit.factor.map(self.lower_bounds_)
        audit["clip_upper"] = audit.factor.map(self.upper_bounds_)
        self.audit_ = audit
        return self

    def selected_features(self, config: SelectionConfig):
        if config.fixed_features is not None:
            return list(config.fixed_features)
        names = self.independent_features_ if config.remove_redundancy else self.ranked_features_
        return names if config.max_factors is None else names[:config.max_factors]

    def report(self, config: SelectionConfig):
        result = self.audit_.copy()
        selected = set(self.selected_features(config))
        result["selected"] = result.factor.isin(selected)
        result["drop_reason"] = result.quality_reason
        good = result.quality_reason.eq("eligible")
        result.loc[good, "drop_reason"] = "below_selected_count"
        if config.remove_redundancy:
            result.loc[good & result.redundant.fillna(False).astype(bool), "drop_reason"] = "high_correlation"
        result.loc[result.selected, "drop_reason"] = "selected"
        result["selection_config"] = config.name
        result["screen_start"] = self.fit_start_
        result["screen_end"] = self.fit_end_
        result["screen_rows"] = self.n_fit_rows_
        result["score_sample_rows"] = len(self.sample_positions_)
        return result


class SelectedColumns(TransformerMixin, BaseEstimator):
    """把训练期已经确定的列和缩尾边界封装到保存的模型内；预测时不重新筛选。"""

    def __init__(self, columns, lower_bounds, upper_bounds):
        self.columns = columns
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def fit(self, X, y=None):
        missing = set(self.columns) - set(X.columns)
        if missing:
            raise ValueError(f"模型输入缺少已选因子：{sorted(missing)}")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        return self

    def transform(self, X):
        values = X.loc[:, self.columns].astype("float32").replace([np.inf, -np.inf], np.nan)
        return values.clip(pd.Series(self.lower_bounds), pd.Series(self.upper_bounds), axis=1)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns, dtype=object)


def selection_configs(screen, counts=(20, 40, 60, 80), pca_variances=(.90, .95)):
    """固定比较清洗基准、Top60、Top120与Top60 PCA95，减少小组作业运行时间。"""
    n = len(screen.independent_features_)
    configs = [SelectionConfig("quality_baseline", None, None, False)]
    if n >= 1:
        configs.append(SelectionConfig("top_60", min(60, n)))
        configs.append(SelectionConfig("top_120", min(120, n)))
        configs.append(SelectionConfig("top_60_pca_95", min(60, n), .95))
    # 少于60维时去重，防止候选方案重复
    unique = {}
    for config in configs:
        unique[config.name] = config
    return list(unique.values())


def make_preprocessor(screen, config):
    names = screen.selected_features(config)
    if not names:
        raise ValueError("没有可输入模型的因子")
    reducer = PCA(n_components=config.pca_variance, svd_solver="full") if config.pca_variance else "passthrough"
    return Pipeline([
        ("select", SelectedColumns(names, screen.lower_bounds_.reindex(names).to_dict(), screen.upper_bounds_.reindex(names).to_dict())),
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True, copy=False)),
        ("scaler", StandardScaler(copy=False)),
        ("pca", reducer),
    ])


def preprocessing_summary(preprocessor, screen, config):
    reducer = preprocessor.named_steps["pca"]
    pca = reducer != "passthrough"
    count = len(screen.selected_features(config))
    return {"selection_config": config.name, "raw_factor_count": len(screen.feature_names_in_),
            "quality_factor_count": len(screen.ranked_features_),
            "independent_factor_count": len(screen.independent_features_), "selected_factor_count": count,
            "model_input_dimensions": int(reducer.n_components_) if pca else count,
            "pca_variance_target": config.pca_variance,
            "pca_explained_variance": float(reducer.explained_variance_ratio_.sum()) if pca else 1.,
            "screen_start": screen.fit_start_, "screen_end": screen.fit_end_, "screen_rows": screen.n_fit_rows_}


def pca_loadings(preprocessor):
    reducer = preprocessor.named_steps["pca"]
    if reducer == "passthrough":
        return pd.DataFrame(columns=["component", "factor", "loading", "explained_variance_ratio"])
    names = preprocessor.named_steps["select"].columns
    return pd.DataFrame([{"component": f"PC{pc + 1}", "factor": name, "loading": float(loading),
                          "explained_variance_ratio": float(reducer.explained_variance_ratio_[pc])}
                         for pc, row in enumerate(reducer.components_) for name, loading in zip(names, row)])


def choose_validation_trial(trials, relative_tolerance=0.):
    """默认严格选验证MAE最小者；仅误差相同时优先更低维。可显式设置容忍度。"""
    if not 0 <= relative_tolerance <= .1:
        raise ValueError("验证MAE相对容忍度必须位于0至0.1")
    valid = [trial for trial in trials if np.isfinite(trial["validation_mae"])]
    if not valid:
        raise ValueError("没有验证指标有效的候选方案")
    best_mae = min(t["validation_mae"] for t in valid)
    eligible = [t for t in valid if t["validation_mae"] <= best_mae * (1 + relative_tolerance) + 1e-12]
    return min(eligible, key=lambda t: (t["model_input_dimensions"], t["selected_factor_count"], t["validation_mae"], t["trial_id"]))
