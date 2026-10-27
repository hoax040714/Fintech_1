# 修改记录：接入417因子筛选、十股时间序列验证、14模型调参和横截面流程评价；支持按周期合并输出与有限CPU线程。
"""模型定义、时间切分、训练、测试评估与最新预测。"""

import hashlib
import json
import os
from pathlib import Path
import sys
import time

# 默认限制数值库为单线程，避免集群上单个任务占满CPU。
for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from price_source import MODEL_DIR, RESULT_DIR, PRICE_FILE, FEATURE_FILE, prepare_folders, source_signature
from factors import FEATURES, HORIZONS, LEGACY_FEATURES, factor_catalog
from .factor_selection import (
    PREPROCESSING_VERSION, FactorScreen, SelectionConfig, selection_configs, make_preprocessor,
    preprocessing_summary, pca_loadings, choose_validation_trial,
)


RANDOM_STATE = 42
SAVED_MODEL_DIR = MODEL_DIR.parent / "saved_models"
MIN_CROSS_SECTION = 5
CPU_THREADS = max(1, int(os.environ.get("FINTECH_CPU_THREADS", "1")))


def save_horizon_csv(frame: pd.DataFrame, filename: str) -> None:
    """按周期增量保存结果；同周期重跑时用新结果替换旧结果。"""
    path = RESULT_DIR / filename
    if "horizon" in frame and path.is_file():
        try:
            old = pd.read_csv(path)
            old = old[~old.horizon.isin(frame.horizon.unique())]
            frame = pd.concat([old, frame], ignore_index=True, sort=False)
        except (ValueError, pd.errors.EmptyDataError):
            pass
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def linear_pipeline(model) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def build_models(modern: bool = False, model_device: str = "cpu") -> dict[str, list[tuple[str, object]]]:
    """构建老师要求的十类传统基线及验证集小网格候选。"""
    from itertools import product
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.linear_model import ElasticNet, HuberRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    import lightgbm as lgb
    import xgboost as xgb

    def raw(estimator):
        return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("model", estimator)])

    def scaled(estimator):
        return linear_pipeline(estimator)

    models = {
        "ols": [(f"alpha={a:g}", scaled(Ridge(alpha=a))) for a in (1e-4, 1e-2, 1.)],
        "ols_huber": [(f"epsilon={e},alpha={a:g}", scaled(HuberRegressor(epsilon=e, alpha=a, max_iter=300)))
                      for e, a in product((1.1, 1.35), (1e-4, 1e-3))],
        "ols3": [(f"alpha={a:g}", scaled(Ridge(alpha=a))) for a in (1e-4, 1e-2)],
        "pls": [(f"n_components={k}", scaled(PLSRegression(n_components=k))) for k in (1, 2, 4, 8)],
        "lasso": [(f"alpha={a:g}", scaled(Lasso(alpha=a, max_iter=10000))) for a in (1e-4, 5e-4, 1e-3, 5e-3)],
        "enet": [(f"alpha={a:g},l1_ratio={r}", scaled(ElasticNet(alpha=a, l1_ratio=r, max_iter=10000)))
                 for a, r in product((1e-4, 1e-3, 5e-3), (.3, .7))],
        "gbrt": [(f"learning_rate={lr},trees={n},depth={d}", raw(GradientBoostingRegressor(
                    loss="huber", learning_rate=lr, n_estimators=n, max_depth=d, random_state=RANDOM_STATE)))
                 for lr, n, d in product((.03, .08), (80, 160), (1, 2, 3))],
        "random_forest": [(f"trees={n},max_features={mf},depth={d},leaf={leaf},ccp_alpha={ccp}",
                    raw(RandomForestRegressor(n_estimators=n, max_features=mf, max_depth=d,
                         min_samples_leaf=leaf, ccp_alpha=ccp, n_jobs=CPU_THREADS, random_state=RANDOM_STATE)))
                 for n, mf, d, leaf, ccp in product((100, 200), (.3, .7), (6, 12), (5, 20), (0., .001))],
        "lightgbm": [(f"leaves={leaves},trees={n},min_child={child},lambda={reg}", raw(lgb.LGBMRegressor(
                    num_leaves=leaves, n_estimators=n, min_child_samples=child, reg_lambda=reg,
                    n_jobs=CPU_THREADS, verbosity=-1, random_state=RANDOM_STATE)))
                 for leaves, n, child, reg in product((7, 15, 31), (100, 200), (20, 50), (0., 1.))],
        "xgboost": [(f"depth={d},trees={n},min_child_weight={w},lambda={reg}", raw(xgb.XGBRegressor(
                    max_depth=d, n_estimators=n, min_child_weight=w, reg_lambda=reg,
                    n_jobs=CPU_THREADS, objective="reg:squarederror", tree_method="hist", verbosity=0,
                    random_state=RANDOM_STATE)))
                 for d, n, w, reg in product((2, 4, 6), (100, 200), (1, 5), (0., 1.))],
    }
    if modern:
        import inspect
        from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Regressor
        from tabpfn import TabPFNRegressor
        realmlp_trials = []
        for epochs, rate in product((30, 60), (1e-3, 3e-3)):
            wanted = {"n_threads": CPU_THREADS, "device": model_device, "random_state": RANDOM_STATE,
                      "n_epochs": epochs, "lr": rate, "batch_size": 128, "verbosity": 0}
            params = {k: v for k, v in wanted.items() if k in inspect.signature(RealMLP_TD_Regressor).parameters}
            realmlp_trials.append((f"epochs={epochs},learning_rate={rate}", raw(RealMLP_TD_Regressor(**params))))
        tab_trials = []
        for estimators in (4, 8):
            cached_v2 = Path.home() / ".cache" / "tabpfn" / "tabpfn-v2-regressor.ckpt"
            wanted = {"n_estimators": estimators, "device": model_device, "random_state": RANDOM_STATE,
                      "n_preprocessing_jobs": 1, "show_progress_bar": False,
                      "ignore_pretraining_limits": True,
                      "model_path": str(cached_v2) if cached_v2.is_file() else "auto"}
            params = {k: v for k, v in wanted.items() if k in inspect.signature(TabPFNRegressor).parameters}
            tab_trials.append((f"n_estimators={estimators}", raw(TabPFNRegressor(**params))))
        models["realmlp"] = realmlp_trials
        models["tabpfn"] = tab_trials
    return models


def chronological_split(
    data: pd.DataFrame, horizon: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按日期做70%/15%/15%切分，并在边界留出标签隔离带。"""
    dates = np.array(sorted(data["date"].dropna().unique()))
    cut1 = int(len(dates) * 0.70)
    cut2 = int(len(dates) * 0.85)

    if cut1 <= horizon or cut2 - cut1 <= horizon or cut2 >= len(dates):
        raise ValueError(f"{horizon}日预测的交易日期不足，无法同时保留训练/验证/测试及标签隔离带")

    train_dates = set(dates[: cut1 - horizon])
    validation_dates = set(dates[cut1 : cut2 - horizon])
    test_dates = set(dates[cut2:])

    train = data[data["date"].isin(train_dates)].copy()
    validation = data[data["date"].isin(validation_dates)].copy()
    test = data[data["date"].isin(test_dates)].copy()
    # 停牌等情形下，N条个股观察可能跨越更多A股日期；再用真实标签终点检查边界。
    if "_label_end_date" in data:
        train = train[train["_label_end_date"] < dates[cut1]].copy()
        validation = validation[validation["_label_end_date"] < dates[cut2]].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("标签隔离后存在空分区，不能训练或选择因子")
    return train, validation, test


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    denominator = float(np.square(y_true).sum())
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "r2_oos_zero": float(1. - np.square(y_true - y_pred).sum() / denominator) if denominator else float("nan"),
        "direction_accuracy": float(np.mean(np.sign(y_true) == np.sign(y_pred))),
        "prediction_correlation": float(pd.Series(y_true).corr(pd.Series(y_pred))),
    }


def file_sha256(path: Path) -> str:
    """分块计算输入文件指纹，避免一次性把大型Parquet读入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_importance(model: Pipeline) -> np.ndarray:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "coef_"):
        values = np.asarray(estimator.coef_, dtype=float).reshape(-1)
    elif hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float).reshape(-1)
    else:
        # 现代表格模型不一定暴露系数/树重要性；结果留0并在method列标记不可用。
        return np.zeros(len(FEATURES), dtype=float)
    if "preprocess" not in model.named_steps:
        return values
    prep = model.named_steps["preprocess"]
    names = prep.named_steps["select"].columns
    reducer = prep.named_steps["pca"]
    if reducer != "passthrough":
        # 线性系数可以回投影；树模型这里只是平方载荷分摊代理，不能解释成原始重要性。
        matrix = reducer.components_.T
        values = matrix @ values if hasattr(estimator, "coef_") else (matrix ** 2) @ values
    output = pd.Series(0., index=FEATURES)
    output.loc[names] = values
    return output.to_numpy()


def importance_method(model):
    linear = hasattr(model.named_steps["model"], "coef_")
    if not linear and not hasattr(model.named_steps["model"], "feature_importances_"):
        return "not_available"
    prep = model.named_steps.get("preprocess")
    pca = prep is not None and prep.named_steps["pca"] != "passthrough"
    return ("pca_backprojected_standardized_coefficient" if pca else "standardized_coefficient") if linear else (
        "pca_squared_loading_proxy" if pca else "impurity_importance")


def horizon_samples(features, horizon):
    """标签数值不变，补充真实终点用于防止个股停牌造成标签跨分区。"""
    target = f"future_return_{horizon}d"
    ordered = features[["date", "code"]].sort_values(["code", "date"])
    ends = ordered.groupby("code", sort=False)["date"].shift(-horizon)
    usable = features.dropna(subset=[target]).copy()
    usable["_label_end_date"] = ends.reindex(usable.index)
    return usable.dropna(subset=["_label_end_date"])


def tune_models(train, validation, target, screen, counts=(60, 120),
                pca_variances=(.95,), relative_tolerance=0., modern=False, model_device="cpu"):
    """预处理每个候选只拟合一次，然后供所有模型参数复用；测试集不是本函数参数。"""
    families = build_models(modern=modern, model_device=model_device)
    all_trials = []
    configs = selection_configs(screen, counts, pca_variances)
    configs.append(SelectionConfig("ols3_fixed", 3, None, False,
                                   ("ext_log_float_cap", "ext_log_pb", "momentum_20d")))
    for config in configs:
        prep = make_preprocessor(screen, config)
        x_train = prep.fit_transform(train[FEATURES])
        x_valid = prep.transform(validation[FEATURES])
        details = preprocessing_summary(prep, screen, config)
        print(f"  验证 {config.name}：{details['selected_factor_count']}因子 → {details['model_input_dimensions']}维", flush=True)
        for model_name, candidates in families.items():
            if model_name == "ols3" and config.name != "ols3_fixed":
                continue
            if model_name != "ols3" and config.name == "ols3_fixed":
                continue
            # 树模型不需要高维/PCA搜索；只比较全量质量因子和Top120，控制运行成本。
            if model_name in {"gbrt", "random_forest", "lightgbm", "xgboost"} and config.name not in {"quality_baseline", "top_120"}:
                continue
            if model_name == "realmlp" and config.name not in {"quality_baseline", "top_120"}:
                continue
            if model_name == "tabpfn" and config.name != "top_60":
                continue
            for label, pipeline in candidates:
                estimator = clone(pipeline.named_steps["model"])
                fit_x, fit_y = x_train, train[target]
                if model_name == "tabpfn" and len(fit_y) > 8000:
                    sampled = train.sample(n=4000, random_state=RANDOM_STATE).sort_index()
                    positions = train.index.get_indexer(sampled.index)
                    fit_x, fit_y = x_train[positions], sampled[target]
                estimator.fit(fit_x, fit_y)
                prediction = np.concatenate([estimator.predict(x_valid[i:i + 32]) for i in range(0, len(x_valid), 32)]) if model_name == "tabpfn" else estimator.predict(x_valid)
                all_trials.append({"model": model_name, "params": label,
                                   "trial_id": f"{model_name}|{config.name}|{label}",
                                   "validation_mae": float(mean_absolute_error(validation[target], prediction)),
                                   "validation_rmse": float(np.sqrt(mean_squared_error(validation[target], prediction))),
                                   "validation_direction_accuracy": float(np.mean(np.sign(prediction) == np.sign(validation[target]))),
                                   **details, "config": config, "estimator": clone(estimator)})
        del x_train, x_valid
    winners = {name: choose_validation_trial([row for row in all_trials if row["model"] == name], relative_tolerance)
               for name in families}
    return winners, all_trials


def fit_selected_model(data, target, screen, config, estimator):
    prep = make_preprocessor(screen, config)
    values = prep.fit_transform(data[FEATURES])
    fitted = clone(estimator).fit(values, data[target])
    # 已拟合的筛选边界、填充、缩放、PCA和回归器一起保存，预测时只做transform。
    return Pipeline([("preprocess", prep), ("model", fitted)])


def _nn_sequences(frame, values, target, window=20):
    """把预处理后的因子按股票分组转为20日序列，返回序列、标签和行位置。"""
    xs, ys, rows = [], [], []
    y = frame[target].to_numpy(dtype=np.float32)
    for _, positions in frame.groupby("code", sort=False).indices.items():
        positions = np.asarray(positions, dtype=int)
        for end in range(window - 1, len(positions)):
            pos = positions[end]
            if np.isfinite(y[pos]):
                xs.append(values[positions[end - window + 1:end + 1]])
                ys.append(y[pos])
                rows.append(pos)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), np.asarray(rows, dtype=int)


def _train_nn(model_name, x_train, y_train, x_valid, y_valid, hidden, learning_rate,
              max_epochs=30, patience=5, seed=RANDOM_STATE):
    """训练CPU/GPU轻量序列网络，早停仅使用验证集，并返回验证预测和最佳轮数。"""
    import torch
    from torch import nn
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(x_train.shape[-1], hidden)
            if model_name == "lstm":
                self.encoder = nn.LSTM(hidden, hidden, batch_first=True)
            else:
                layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=4,
                    dim_feedforward=hidden * 2, batch_first=True, dropout=.1)
                self.encoder = nn.TransformerEncoder(layer, num_layers=1)
                self.position = nn.Parameter(torch.zeros(1, x_train.shape[1], hidden))
            self.head = nn.Linear(hidden, 1)
        def forward(self, x):
            x = self.proj(x)
            if model_name == "transformer":
                x = x + self.position
            x = self.encoder(x)
            if isinstance(x, tuple):
                x = x[0]
            return self.head(x[:, -1]).squeeze(-1)

    device = torch.device("cpu")
    model = Net().to(device)
    center, scale = float(y_train.mean()), float(y_train.std()) or 1.
    yt = torch.as_tensor((y_train - center) / scale, dtype=torch.float32)
    yv = torch.as_tensor((y_valid - center) / scale, dtype=torch.float32)
    xt = torch.as_tensor(x_train, dtype=torch.float32)
    xv = torch.as_tensor(x_valid, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    best_loss, best_epoch, wait, state = float("inf"), 1, 0, None
    for epoch in range(1, max_epochs + 1):
        model.train()
        # 小批次、单线程；无随机打乱也能避免跨窗口样本的资源峰值。
        for start in range(0, len(xt), 128):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xt[start:start + 128]), yt[start:start + 128])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(xv), yv).item())
        if val_loss < best_loss:
            best_loss, best_epoch, wait = val_loss, epoch, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        prediction = model(xv).cpu().numpy() * scale + center
    return prediction, best_epoch


def _refit_nn(model_name, x_train, y_train, x_test, hidden, learning_rate, epochs,
              seeds=3):
    """按验证阶段选出的轮数在训练+验证数据上重训，并平均多个固定种子预测。"""
    import torch
    from torch import nn
    predictions = []
    for seed in range(RANDOM_STATE, RANDOM_STATE + seeds):
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(x_train.shape[-1], hidden)
                if model_name == "lstm":
                    self.encoder = nn.LSTM(hidden, hidden, batch_first=True)
                else:
                    layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=4,
                        dim_feedforward=hidden * 2, batch_first=True, dropout=.1)
                    self.encoder = nn.TransformerEncoder(layer, num_layers=1)
                    self.position = nn.Parameter(torch.zeros(1, x_train.shape[1], hidden))
                self.head = nn.Linear(hidden, 1)
            def forward(self, x):
                x = self.proj(x)
                if model_name == "transformer": x = x + self.position
                x = self.encoder(x)
                if isinstance(x, tuple): x = x[0]
                return self.head(x[:, -1]).squeeze(-1)
        model = Net()
        center, scale = float(y_train.mean()), float(y_train.std()) or 1.
        xt = torch.as_tensor(x_train, dtype=torch.float32)
        yt = torch.as_tensor((y_train - center) / scale, dtype=torch.float32)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        for _ in range(epochs):
            model.train()
            for start in range(0, len(xt), 128):
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(xt[start:start + 128]), yt[start:start + 128])
                loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            predictions.append(model(torch.as_tensor(x_test, dtype=torch.float32)).numpy() * scale + center)
    return np.mean(predictions, axis=0)


def evaluate_panel_predictions(predictions: pd.DataFrame):
    """输出逐日IC、逐股误差和5v5流程验证；10只股票每日IC标准误约0.3、约240次调仓，不能作投资结论。"""
    daily_rows, stock_rows, nav_rows, summary_rows = [], [], [], []
    for (model, horizon), part in predictions.groupby(["model", "horizon"]):
        for date, day in part.groupby("date", sort=True):
            day = day.dropna(subset=["actual_return", "predicted_return"])
            if len(day) >= 5:
                daily_rows.append({"model": model, "horizon": horizon, "date": date,
                    "n_stocks": len(day), "pearson_ic": day.actual_return.corr(day.predicted_return),
                    "rank_ic": day.actual_return.rank().corr(day.predicted_return.rank())})
        for code, stock in part.groupby("code"):
            stock_rows.append({"model": model, "horizon": horizon, "code": code,
                "n_test": len(stock), "mae": mean_absolute_error(stock.actual_return, stock.predicted_return),
                "correlation": stock.actual_return.corr(stock.predicted_return) if len(stock) > 1 else np.nan})

        # 5v5组合每h个交易日调仓；10只股票的每日截面IC误差较大，只作流程验证。
        dates = sorted(part.date.unique())
        prior_ls, prior_long = {}, {}
        for date in dates[::int(horizon)]:
            day = part[part.date == date].dropna(subset=["actual_return", "predicted_return"])
            if len(day) < 10:
                continue
            ranked = day.sort_values("predicted_return")
            shorts = ranked.head(5).code.astype(str).tolist()
            longs = ranked.tail(5).code.astype(str).tolist()
            weights = {code: .2 for code in longs} | {code: -.2 for code in shorts}
            turnover_ls = .5 * sum(abs(weights.get(code, 0.) - prior_ls.get(code, 0.))
                                   for code in set(weights) | set(prior_ls))
            prior_ls = weights
            long_ret = float(ranked.tail(5).actual_return.mean())
            short_ret = float(ranked.head(5).actual_return.mean())
            all_ret = float(day.actual_return.mean())
            long_weights = {code: .2 for code in longs}
            turnover_long = .5 * sum(abs(long_weights.get(code, 0.) - prior_long.get(code, 0.))
                                     for code in set(long_weights) | set(prior_long))
            prior_long = long_weights
            for strategy, gross, turnover in (("long_short_5v5", long_ret - short_ret, turnover_ls),
                                              ("long_top5_vs_equal_weight", long_ret - all_ret, turnover_long)):
                for bps in (10, 20):
                    cost = (bps / 10000) * turnover
                    net = gross - cost
                    nav_rows.append({"model": model, "horizon": horizon, "date": date,
                        "strategy": strategy, "cost_bps": bps, "gross_return": gross,
                        "turnover": turnover, "cost": cost, "net_return": net})
    daily = pd.DataFrame(daily_rows)
    ic_rows = []
    if not daily.empty:
        for (model, horizon), part in daily.groupby(["model", "horizon"]):
            for metric in ("pearson_ic", "rank_ic"):
                values = part[metric].dropna()
                std = values.std(ddof=1)
                ic_rows.append({"model": model, "horizon": horizon, "metric": metric,
                    "mean": values.mean(), "std": std,
                    "icir": values.mean() / std if std and np.isfinite(std) else np.nan,
                    "t_stat": values.mean() / std * np.sqrt(len(values)) if std and np.isfinite(std) else np.nan,
                    "positive_ratio": (values > 0).mean(), "n_days": len(values)})
    nav = pd.DataFrame(nav_rows)
    if not nav.empty:
        for keys, group in nav.groupby(["model", "horizon", "strategy", "cost_bps"]):
            model, horizon, strategy, bps = keys
            returns = group.net_return.to_numpy()
            curve = np.cumprod(1 + returns)
            drawdown = curve / np.maximum.accumulate(curve) - 1
            std = returns.std(ddof=1)
            summary_rows.append({"model": model, "horizon": horizon, "strategy": strategy,
                "cost_bps": bps, "periods": len(returns), "cumulative_return": curve[-1] - 1,
                "annualized_sharpe": returns.mean() / std * np.sqrt(252 / horizon) if std else np.nan,
                "max_drawdown": drawdown.min(), "average_turnover": group.turnover.mean()})
        nav["nav"] = nav.groupby(["model", "horizon", "strategy", "cost_bps"]).net_return.transform(lambda x: (1 + x).cumprod())
        initial_rows = []
        for keys, group in nav.groupby(["model", "horizon", "strategy", "cost_bps"]):
            model, horizon, strategy, bps = keys
            row = group.iloc[0].to_dict()
            row.update({"date": pd.Timestamp(group.date.min()) - pd.Timedelta(days=1),
                        "gross_return": 0., "turnover": 0., "cost": 0., "net_return": 0., "nav": 1.})
            initial_rows.append(row)
        nav = pd.concat([nav, pd.DataFrame(initial_rows)], ignore_index=True).sort_values(
            ["model", "horizon", "strategy", "cost_bps", "date"])
    save_horizon_csv(daily, "ic_daily.csv")
    save_horizon_csv(pd.DataFrame(ic_rows), "ic_summary.csv")
    save_horizon_csv(pd.DataFrame(stock_rows), "per_stock_metrics.csv")
    save_horizon_csv(nav, "backtest_nav.csv")
    save_horizon_csv(pd.DataFrame(summary_rows), "backtest_summary.csv")


def run_factor_group_ablation(features: pd.DataFrame, horizons=(1, 5)):
    """用训练期拟合的固定Ridge比较新因子组；结果只用于流程验证。"""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    catalog = factor_catalog().set_index("factor")
    type_groups = {
        "量价": ["价格类", "成交量与流动性类", "波动率类", "动量类", "反转类", "Alpha组合类"],
        "市场行业": ["市场与行业类"],
        "基本面": ["盈利质量类", "成长类", "估值与规模类", "信息时效与质量类", "行业相对基本面类"],
        "宏观": ["宏观利率类", "宏观增长类"],
        "日历": ["日历效应类"], "全球市场": ["全球市场类"],
        "交互/形态": ["市场形态扩展类", "跨类别交互类"],
    }
    group_features = {name: catalog.index[catalog.factor_type.isin(types)].tolist()
                      for name, types in type_groups.items()}
    order = list(type_groups)
    combinations = [(name, group_features[name]) for name in order]
    running = []
    for name in order:
        running.extend(group_features[name])
        combinations.append(("累积:" + "→".join(order[:order.index(name) + 1]), running.copy()))
    combinations += [("全部417", list(FEATURES)), ("原有137", list(LEGACY_FEATURES))]
    rows = []
    for horizon in horizons:
        target = f"future_return_{horizon}d"
        train, valid, test = chronological_split(horizon_samples(features, horizon), horizon)
        for label, columns in combinations:
            columns = [c for c in columns if c in train and train[c].notna().any()]
            model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                  StandardScaler(), Ridge(alpha=1.))
            model.fit(train[columns], train[target])
            pred_valid = model.predict(valid[columns])
            pred_test = model.predict(test[columns])
            denominator = float(np.square(test[target].to_numpy()).sum())
            rows.append({"horizon": horizon, "group": label, "n_factors": len(columns),
                "validation_mae": mean_absolute_error(valid[target], pred_valid),
                "test_mae": mean_absolute_error(test[target], pred_test),
                "r2_oos_zero": 1. - np.square(test[target].to_numpy() - pred_test).sum() / denominator if denominator else np.nan,
                "direction_accuracy": np.mean(np.sign(test[target]) == np.sign(pred_test))})
    output = pd.DataFrame(rows)
    save_horizon_csv(output, "factor_group_ablation.csv")
    return output


def train_models(features: pd.DataFrame, relative_tolerance: float = 0., neural: bool = False,
                 modern: bool = False, model_device: str = "cpu", nn_seeds: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """训练期筛选，验证期选择数量/PCA/参数，测试期只评估，成熟全样本另拟合生产模型。"""
    prepare_folders()
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    run_started = time.time()
    run_started_at = pd.Timestamp.now().isoformat(timespec="seconds")
    if not 0 <= relative_tolerance <= .1:
        raise ValueError("验证MAE相对容忍度必须位于0至0.1")
    metric_rows = []
    importance_rows = []
    prediction_rows = []
    latest_rows = []
    trial_rows, selection_rows, preprocessing_rows, loading_rows = [], [], [], []
    selected_feature_rows = []
    production_inputs = {}

    latest_date = features["date"].max()
    latest = features[features["date"] == latest_date].dropna(subset=FEATURES, how="all").copy()

    for horizon in HORIZONS:
        target = f"future_return_{horizon}d"
        usable = horizon_samples(features, horizon)
        train, validation, test = chronological_split(usable, horizon)
        print(f"筛选与验证 / {horizon}日 ...", flush=True)
        screen = FactorScreen(min_cross_section=MIN_CROSS_SECTION).fit(train[FEATURES], train[target], train["date"])
        winners, trials = tune_models(train, validation, target, screen, relative_tolerance=relative_tolerance,
                                      modern=modern, model_device=model_device)
        for row in trials:
            trial_rows.append({"horizon": horizon, **{k: v for k, v in row.items() if k not in ("config", "estimator")},
                               "chosen": winners[row["model"]]["trial_id"] == row["trial_id"]})
        train_validation = pd.concat([train, validation], ignore_index=True)
        evaluation_screen = FactorScreen(min_cross_section=MIN_CROSS_SECTION).fit(train_validation[FEATURES], train_validation[target], train_validation["date"])

        for model_name, winner in winners.items():
            config = winner["config"]
            print(f"训练 {model_name} / {horizon} 日：{config.name} / {winner['params']}", flush=True)
            fit_frame = train_validation
            estimator = clone(winner["estimator"])
            if model_name == "tabpfn" and len(fit_frame) > 8000:
                fit_frame = fit_frame.sample(n=4000, random_state=RANDOM_STATE).sort_values(["date", "code"])
            best_model = fit_selected_model(fit_frame, target, evaluation_screen, config, estimator)
            if model_name == "tabpfn":
                test_pred = np.concatenate([best_model.predict(test[FEATURES].iloc[i:i + 32]) for i in range(0, len(test), 32)])
            else:
                test_pred = best_model.predict(test[FEATURES])

            metric_rows.append(
                {
                    "model": model_name,
                    "horizon": horizon,
                    "best_params": winner["params"],
                    "validation_mae": winner["validation_mae"],
                    "mae_tolerance": relative_tolerance,
                    **preprocessing_summary(best_model.named_steps["preprocess"], evaluation_screen, config),
                    "train_start": train["date"].min(),
                    "train_end": train["date"].max(),
                    "validation_start": validation["date"].min(),
                    "validation_end": validation["date"].max(),
                    "test_start": test["date"].min(),
                    "test_end": test["date"].max(),
                    "n_train": len(train),
                    "n_validation": len(validation),
                    "n_test": len(test),
                    **regression_metrics(test[target], test_pred),
                }
            )

            predictions = test[["date", "code", "name", target]].copy()
            predictions = predictions.rename(columns={target: "actual_return"})
            predictions["predicted_return"] = test_pred
            predictions["model"] = model_name
            predictions["horizon"] = horizon
            prediction_rows.append(predictions)

            for factor, value in zip(FEATURES, feature_importance(best_model)):
                importance_rows.append(
                    {
                        "model": model_name,
                        "horizon": horizon,
                        "factor": factor,
                        "importance": float(value),
                        "importance_method": importance_method(best_model),
                        "stage": "test_evaluation",
                    }
                )

            for stage, fitted_screen, fitted_model in (
                ("search_train", screen, None), ("test_evaluation", evaluation_screen, best_model),
            ):
                audit = fitted_screen.report(config)
                audit["model"], audit["horizon"], audit["stage"] = model_name, horizon, stage
                selection_rows.append(audit)
                if fitted_model is not None:
                    prep = fitted_model.named_steps["preprocess"]
                    preprocessing_rows.append({"model": model_name, "horizon": horizon, "stage": stage,
                                               **preprocessing_summary(prep, fitted_screen, config)})
                    loadings = pca_loadings(prep)
                    loadings["model"], loadings["horizon"], loadings["stage"] = model_name, horizon, stage
                    loading_rows.append(loadings)

        # 本周期所有测试评估结束后，才允许使用全部成熟标签拟合最新预测模型。
        if neural:
            # 序列模型只比较Top60；每只股票独立构造20日窗口，验证阶段只按验证MAE选超参。
            nn_config = next(c for c in selection_configs(screen) if c.name == "top_60")
            prep = make_preprocessor(screen, nn_config)
            xtr = prep.fit_transform(train[FEATURES])
            xva = prep.transform(validation[FEATURES])
            best_nn = {}
            for model_name in ("lstm", "transformer"):
                choices = []
                for hidden in (16, 32):
                    for rate in (5e-4, 1e-3):
                        tr_x, tr_y, _ = _nn_sequences(train.reset_index(drop=True), xtr, target)
                        va_x, va_y, va_rows = _nn_sequences(validation.reset_index(drop=True), xva, target)
                        prediction, epochs = _train_nn(model_name, tr_x, tr_y, va_x, va_y,
                                                       hidden, rate, seed=RANDOM_STATE)
                        mae = float(mean_absolute_error(va_y, prediction))
                        params = f"hidden={hidden},learning_rate={rate:g},best_epoch={epochs}"
                        trial_id = f"{model_name}|top_60|{params}"
                        choices.append({"model": model_name, "params": params, "trial_id": trial_id,
                                        "validation_mae": mae, "validation_rmse": float(np.sqrt(mean_squared_error(va_y, prediction))),
                                        "config": nn_config, "hidden": hidden, "learning_rate": rate, "epochs": epochs})
                winner = min(choices, key=lambda row: row["validation_mae"])
                best_nn[model_name] = winner
                for row in choices:
                    trial_rows.append({"horizon": horizon, **{k: v for k, v in row.items() if k != "config"},
                                       "chosen": row["trial_id"] == winner["trial_id"]})

            nn_eval_screen = FactorScreen(min_cross_section=MIN_CROSS_SECTION).fit(
                train_validation[FEATURES], train_validation[target], train_validation["date"])
            nn_prep = make_preprocessor(nn_eval_screen, nn_config)
            x_tv = nn_prep.fit_transform(train_validation[FEATURES])
            xte = nn_prep.transform(test[FEATURES])
            tv_x, tv_y, _ = _nn_sequences(train_validation.reset_index(drop=True), x_tv, target)
            te_x, te_y, te_rows = _nn_sequences(test.reset_index(drop=True), xte, target)
            for model_name, winner in best_nn.items():
                pred = _refit_nn(model_name, tv_x, tv_y, te_x, winner["hidden"],
                                 winner["learning_rate"], winner["epochs"], seeds=nn_seeds)
                test_nn = test.reset_index(drop=True).iloc[te_rows].copy()
                metric_rows.append({"model": model_name, "horizon": horizon,
                    "best_params": winner["params"], "validation_mae": winner["validation_mae"],
                    **preprocessing_summary(nn_prep, nn_eval_screen, nn_config),
                    "train_start": train["date"].min(), "train_end": train["date"].max(),
                    "validation_start": validation["date"].min(), "validation_end": validation["date"].max(),
                    "test_start": test_nn.date.min(), "test_end": test_nn.date.max(),
                    "n_train": len(tv_x), "n_validation": len(va_x), "n_test": len(te_x),
                    **regression_metrics(te_y, pred)})
                pred_frame = test_nn[["date", "code", "name", target]].rename(columns={target: "actual_return"})
                pred_frame["predicted_return"] = pred
                pred_frame["model"], pred_frame["horizon"] = model_name, horizon
                prediction_rows.append(pred_frame)
                selected_feature_rows.append({"model": model_name, "horizon": horizon,
                    "feature_plan": nn_config.name, "factors": ",".join(nn_eval_screen.selected_features(nn_config))})

        # 本周期所有测试评估结束后，才允许使用全部成熟标签拟合生产模型。
        production_screen = FactorScreen(min_cross_section=MIN_CROSS_SECTION).fit(usable[FEATURES], usable[target], usable["date"])
        for model_name, winner in winners.items():
            if model_name in {"lstm", "transformer"}:
                continue
            config = winner["config"]
            production_frame = usable
            if model_name == "tabpfn" and len(production_frame) > 8000:
                production_frame = production_frame.sample(n=4000, random_state=RANDOM_STATE).sort_values(["date", "code"])
            production_model = fit_selected_model(production_frame, target, production_screen, config, winner["estimator"])
            joblib.dump(production_model, SAVED_MODEL_DIR / f"{model_name}_{horizon}d.joblib")
            prep = production_model.named_steps["preprocess"]
            summary = preprocessing_summary(prep, production_screen, config)
            production_inputs[f"{model_name}_{horizon}d"] = {
                "selected_factors": production_screen.selected_features(config), **summary,
            }
            selected_feature_rows.append({"model": model_name, "horizon": horizon,
                "feature_plan": config.name, "factors": ",".join(production_screen.selected_features(config))})
            audit = production_screen.report(config)
            audit["model"], audit["horizon"], audit["stage"] = model_name, horizon, "production"
            selection_rows.append(audit)
            preprocessing_rows.append({"model": model_name, "horizon": horizon, "stage": "production", **summary})
            loadings = pca_loadings(prep)
            loadings["model"], loadings["horizon"], loadings["stage"] = model_name, horizon, "production"
            loading_rows.append(loadings)
            latest_prediction = latest[["date", "code", "name", "close"]].copy()
            latest_prediction["predicted_return"] = production_model.predict(latest[FEATURES])
            latest_prediction["reference_target_price"] = (
                latest_prediction["close"] * (1 + latest_prediction["predicted_return"])
            )
            latest_prediction["model"] = model_name
            latest_prediction["horizon"] = horizon
            latest_rows.append(latest_prediction)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    evaluate_panel_predictions(predictions)
    ablation_horizons = tuple(h for h in HORIZONS if h in (1, 5))
    if ablation_horizons:
        run_factor_group_ablation(features, ablation_horizons)

    save_horizon_csv(metrics, "model_metrics.csv")
    save_horizon_csv(predictions, "test_predictions.csv")
    save_horizon_csv(pd.DataFrame(importance_rows), "feature_importance.csv")
    save_horizon_csv(pd.concat(latest_rows, ignore_index=True), "latest_predictions.csv")
    save_horizon_csv(pd.DataFrame(trial_rows), "model_selection_trials.csv")
    save_horizon_csv(pd.DataFrame(trial_rows), "tuning_trials.csv")
    save_horizon_csv(pd.concat(selection_rows, ignore_index=True), "factor_selection.csv")
    save_horizon_csv(pd.DataFrame(preprocessing_rows), "preprocessing_summary.csv")
    save_horizon_csv(pd.concat(loading_rows, ignore_index=True), "pca_loadings.csv")
    save_horizon_csv(pd.DataFrame(selected_feature_rows), "selected_features.csv")

    metadata = {
        "data_source": "data/features.parquet",
        "preprocessing_version": PREPROCESSING_VERSION,
        "model_inputs": production_inputs,
        "source_sha256": source_signature(),
        "feature_source_sha256": file_sha256(FEATURE_FILE),
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "latest_market_date": str(pd.Timestamp(latest_date).date()),
        "features": FEATURES,
        "horizons": list(HORIZONS),
        "stock_count": int(features["code"].nunique()),
        "stock_codes": sorted(features["code"].astype(str).unique().tolist()),
        "feature_count": len(FEATURES), "min_cross_section": MIN_CROSS_SECTION,
        "device": model_device, "thread_limit": CPU_THREADS,
        "neural_enabled": neural, "modern_enabled": modern, "nn_seeds": nn_seeds,
        "python": sys.version.split()[0], "run_started_at": run_started_at,
        "run_finished_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.time() - run_started, 2),
        "command": " ".join(sys.argv),
        "gpu_name": None,
    }
    if modern and model_device == "cuda":
        try:
            import torch
            metadata["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        except Exception:
            metadata["gpu_name"] = "unavailable"
    metadata_path = RESULT_DIR / "run_metadata.json"
    if metadata_path.is_file():
        try:
            previous = json.loads(metadata_path.read_text(encoding="utf-8"))
            current_record = {key: metadata[key] for key in (
                "horizons", "run_started_at", "run_finished_at", "elapsed_seconds",
                "command", "device", "gpu_name", "thread_limit")}
            previous_history = previous.get("run_history") or [{key: previous[key] for key in (
                "horizons", "run_started_at", "run_finished_at", "elapsed_seconds",
                "command", "device", "gpu_name", "thread_limit") if key in previous}]
            metadata["run_history"] = previous_history + [current_record]
            metadata["horizons"] = sorted(set(previous.get("horizons", [])) | set(metadata["horizons"]))
            metadata["model_inputs"] = {**previous.get("model_inputs", {}), **metadata["model_inputs"]}
            metadata["run_started_at"] = previous.get("run_started_at", metadata["run_started_at"])
            metadata["elapsed_seconds"] = round(
                float(previous.get("elapsed_seconds", 0)) + float(metadata["elapsed_seconds"]), 2)
        except (ValueError, KeyError, TypeError):
            pass
    encoded_metadata = json.dumps(metadata, ensure_ascii=False, indent=2, default=str)
    (RESULT_DIR / "metadata.json").write_text(encoded_metadata, encoding="utf-8")
    (RESULT_DIR / "run_metadata.json").write_text(encoded_metadata, encoding="utf-8")
    return metrics, predictions


def audit_factor_selection(features: pd.DataFrame):
    """只检查各预测周期的训练期筛选和候选维数；不选择验证赢家，不生成预测。"""
    folder = RESULT_DIR / "selection_audit"
    folder.mkdir(parents=True, exist_ok=True)
    audits, summaries = [], []
    for horizon in HORIZONS:
        target = f"future_return_{horizon}d"
        train, _, _ = chronological_split(horizon_samples(features, horizon), horizon)
        screen = FactorScreen(min_cross_section=MIN_CROSS_SECTION).fit(train[FEATURES], train[target], train["date"])
        # 诊断PCA只用训练期的按日期均衡样本，完整训练时会重新在完整训练分区拟合。
        sampled = train.iloc[screen.sample_positions_]
        print(f"筛选审计 / {horizon}日：417 → 质量{len(screen.ranked_features_)} → 去冗余{len(screen.independent_features_)}", flush=True)
        for config in selection_configs(screen):
            prep = make_preprocessor(screen, config).fit(sampled[FEATURES])
            summary = preprocessing_summary(prep, screen, config)
            summaries.append({"horizon": horizon, "stage": "training_sample_diagnostic",
                              "pca_fit_rows": len(sampled), **summary})
            audit = screen.report(config)
            audit["horizon"] = horizon
            audits.append(audit)
    pd.concat(audits, ignore_index=True).to_csv(folder / "factor_screening.csv", index=False, encoding="utf-8-sig")
    result = pd.DataFrame(summaries)
    result.to_csv(folder / "dimension_candidates.csv", index=False, encoding="utf-8-sig")
    manifest = {"preprocessing_version": PREPROCESSING_VERSION, "data_source": PRICE_FILE.name,
                "source_sha256": source_signature(), "feature_rows": len(features),
                "stock_count": int(features["code"].nunique()), "horizons": list(HORIZONS),
                "stage": "training_sample_diagnostic", "validation_winner_selected": False}
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
