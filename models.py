"""模型定义、时间切分、训练、测试评估与最新预测。"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data import MODEL_DIR, RESULT_DIR, prepare_folders
from factors import FEATURES, HORIZONS


RANDOM_STATE = 42
MODEL_LABELS = {
    "lasso": "Lasso",
    "ridge": "Ridge",
    "random_forest": "Random Forest",
    "ols": "OLS（弱Ridge）",
    "ols3": "OLS3（双因子）",
    "pls": "PLS",
    "enet": "ElasticNet",
    "gbrt": "GBRT",
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "lstm": "LSTM",
    "transformer": "Transformer",
    "realmlp": "RealMLP",
    "tabpfn": "TabPFN（128条抽样）",
}
DEFAULT_MODELS = list(MODEL_LABELS)[:10]


def linear_pipeline(model) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def build_models(model_names=None, device="cpu") -> dict[str, list[tuple[str, object]]]:
    """在这里增加模型或候选参数。"""
    candidates = {
        "lasso": [
            (f"alpha={alpha:g}", linear_pipeline(Lasso(alpha=alpha, max_iter=10000)))
            for alpha in (0.0001, 0.0005, 0.001)
        ],
        "ridge": [
            (f"alpha={alpha:g}", linear_pipeline(Ridge(alpha=alpha)))
            for alpha in (0.1, 1.0, 10.0)
        ],
        "random_forest": [
            (
                "trees=150,min_leaf=20",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "model",
                            RandomForestRegressor(
                                n_estimators=150,
                                max_depth=12,
                                min_samples_leaf=20,
                                max_features=0.8,
                                n_jobs=-1,
                                random_state=RANDOM_STATE,
                            ),
                        ),
                    ]
                ),
            )
        ],
    }
    # 重复的 Lasso / Random Forest 保留当前项目参数，其余沿用参考目录。
    for name in model_names or DEFAULT_MODELS:
        if name in candidates or name in {"lstm", "transformer"}:
            continue
        if name in {"ols", "ols3"}:
            candidates[name] = [("alpha=0.0001", linear_pipeline(Ridge(alpha=1e-4)))]
        elif name == "pls":
            candidates[name] = [("n_components=2", linear_pipeline(PLSRegression(n_components=2)))]
        elif name == "enet":
            candidates[name] = [("alpha=0.001,l1_ratio=0.5", linear_pipeline(
                ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=3000)
            ))]
        else:
            if name == "gbrt":
                estimator = GradientBoostingRegressor(
                    n_estimators=80, learning_rate=0.05, max_depth=2, random_state=RANDOM_STATE
                )
            elif name == "lgbm":
                from lightgbm import LGBMRegressor
                estimator = LGBMRegressor(
                    n_estimators=100, learning_rate=0.05, max_depth=4, num_leaves=15,
                    n_jobs=1, verbosity=-1, random_state=RANDOM_STATE,
                )
            elif name == "xgb":
                from xgboost import XGBRegressor
                estimator = XGBRegressor(
                    n_estimators=100, learning_rate=0.05, max_depth=3, n_jobs=1,
                    objective="reg:squarederror", tree_method="hist", random_state=RANDOM_STATE,
                )
            elif name == "realmlp":
                from pytabkit import RealMLP_TD_Regressor
                estimator = RealMLP_TD_Regressor(
                    n_threads=1, device=device, random_state=RANDOM_STATE,
                    n_epochs=30, batch_size=128, verbosity=0,
                )
            elif name == "tabpfn":
                import torch
                from tabpfn import TabPFNRegressor
                torch.set_num_threads(1)
                estimator = TabPFNRegressor(
                    n_estimators=4, device=device, random_state=RANDOM_STATE,
                    n_preprocessing_jobs=1, ignore_pretraining_limits=True,
                    model_path="auto",
                )
            candidates[name] = [("参考目录固定参数", Pipeline([
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("model", estimator),
            ]))]
    return {name: candidates[name] for name in (model_names or DEFAULT_MODELS)
            if name not in {"lstm", "transformer"}}


def chronological_split(
    data: pd.DataFrame, horizon: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按日期做70%/15%/15%切分，并在边界留出标签隔离带。"""
    dates = np.array(sorted(data["date"].dropna().unique()))
    cut1 = int(len(dates) * 0.70)
    cut2 = int(len(dates) * 0.85)

    train_dates = set(dates[: cut1 - horizon])
    validation_dates = set(dates[cut1 : cut2 - horizon])
    test_dates = set(dates[cut2:])

    train = data[data["date"].isin(train_dates)].copy()
    validation = data[data["date"].isin(validation_dates)].copy()
    test = data[data["date"].isin(test_dates)].copy()
    return train, validation, test


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "direction_accuracy": float(np.mean(np.sign(y_true.to_numpy()) == np.sign(y_pred))),
        "prediction_correlation": float(pd.Series(y_true.to_numpy()).corr(pd.Series(y_pred))),
    }


def feature_importance(model: Pipeline, X, y, model_name="") -> tuple[np.ndarray, str]:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "coef_"):
        return np.asarray(estimator.coef_, dtype=float).ravel(), "系数"
    if hasattr(estimator, "feature_importances_"):
        return np.asarray(estimator.feature_importances_, dtype=float), "模型内置重要性"
    limit = 32 if model_name == "tabpfn" else 128
    sample = X.sample(n=min(limit, len(X)), random_state=RANDOM_STATE)
    result = permutation_importance(
        model, sample, y.loc[sample.index], scoring="neg_mean_absolute_error",
        n_repeats=1, random_state=RANDOM_STATE,
    )
    return result.importances_mean, "置换重要性（MAE增量）"


def fit_rows(frame, model_name):
    """TabPFN 只从本次允许训练的数据中抽样128行。"""
    if model_name == "tabpfn" and len(frame) > 128:
        return frame.sample(n=128, random_state=RANDOM_STATE).sort_values(["date", "code"])
    return frame


def predict_model(model, X, model_name):
    """TabPFN 每次预测32行，其余模型直接预测。"""
    if model_name == "tabpfn":
        return np.concatenate([
            np.asarray(model.predict(X.iloc[start:start + 32])).ravel()
            for start in range(0, len(X), 32)
        ])
    return np.asarray(model.predict(X)).ravel()


def build_sequence_model(model_name, input_size, window=20):
    """沿用参考目录的小型 LSTM / 单层 Transformer。"""
    from torch import nn
    import torch

    class SequenceModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(input_size, 32)
            if model_name == "lstm":
                self.encoder = nn.LSTM(32, 16, batch_first=True)
                self.head = nn.Linear(16, 1)
            else:
                layer = nn.TransformerEncoderLayer(
                    d_model=32, nhead=4, dim_feedforward=64, dropout=0.1, batch_first=True
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=1)
                self.position = nn.Parameter(torch.zeros(1, window, 32))
                self.head = nn.Linear(32, 1)

        def forward(self, x):
            x = self.projection(x)
            if model_name == "transformer":
                x = self.encoder(x + self.position)
            else:
                x, _ = self.encoder(x)
            return self.head(x[:, -1, :]).squeeze(-1)

    return SequenceModel()


def sequence_data(frame, columns, median, std, window=20):
    """记录每只股票的窗口终点；只在取批次时展开20日窗口，避免占满内存。"""
    frame = frame.sort_values(["code", "date"]).reset_index(drop=True)
    values = ((frame[columns].fillna(median) - median) / std).clip(-10, 10)
    values = values.to_numpy(np.float32)
    ends = []
    for _, stock in frame.groupby("code", sort=False):
        ends.extend(range(stock.index[0] + window - 1, stock.index[-1] + 1))
    ends = np.asarray(ends, dtype=int)
    return values, ends, frame.iloc[ends].copy()


def predict_sequence(model, values, ends, state, device="cpu", batch_size=128):
    """按批次预测，每个窗口只取同一只股票的历史记录。"""
    import torch
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(ends), batch_size):
            positions = ends[start:start + batch_size, None] - np.arange(state["window"] - 1, -1, -1)
            x = torch.from_numpy(values[positions]).to(device)
            predictions.append(model(x).cpu().numpy())
    if not predictions:
        return np.empty(0)
    return np.concatenate(predictions) * state["target_scale"] + state["target_center"]


def fit_sequence(model_name, train, validation, columns, target, epochs=8, device="cpu"):
    """训练期预处理，验证期选择轮数；传入空验证表时按固定轮数重新拟合。"""
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(RANDOM_STATE)
    median = train[columns].median().fillna(0)
    std = train[columns].std().replace(0, 1).fillna(1)
    values, ends, rows = sequence_data(train, columns, median, std)
    target_values = rows[target].to_numpy(np.float32)
    state = {
        "model": model_name, "features": columns, "median": median, "std": std,
        "window": 20, "target_center": float(target_values.mean()),
        "target_scale": float(target_values.std()) or 1.0,
        "fit_rows": len(ends),
    }
    target_values = (target_values - state["target_center"]) / state["target_scale"]
    model = build_sequence_model(model_name, len(columns)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    if not validation.empty:
        valid_values, valid_ends, valid_rows = sequence_data(validation, columns, median, std)
    best_mae = np.inf
    best_state = None
    best_epoch = epochs

    for epoch in range(epochs):
        model.train()
        for start in range(0, len(ends), 128):
            positions = ends[start:start + 128, None] - np.arange(19, -1, -1)
            x = torch.from_numpy(values[positions]).to(device)
            y = torch.from_numpy(target_values[start:start + 128]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(x), y).backward()
            optimizer.step()
        if not validation.empty:
            pred = predict_sequence(model, valid_values, valid_ends, state, device)
            mae = mean_absolute_error(valid_rows[target], pred)
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch + 1
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    state["epochs"] = best_epoch
    return model, state


def predict_saved_sequence(saved, frame, device="cpu"):
    """读取 joblib 字典后恢复网络，返回具备20日窗口的记录和预测。"""
    model = build_sequence_model(saved["model"], len(saved["features"]), saved["window"]).to(device)
    model.load_state_dict(saved["state_dict"])
    values, ends, rows = sequence_data(
        frame, saved["features"], saved["median"], saved["std"], saved["window"]
    )
    return rows, predict_sequence(model, values, ends, saved, device)


def train_models(features: pd.DataFrame, model_names=None, horizons=None,
                 nn_epochs=8, device="cpu") -> tuple[pd.DataFrame, pd.DataFrame]:
    """验证集选参数，测试集评估，全部成熟样本训练最终模型。"""
    prepare_folders()
    metric_rows = []
    importance_rows = []
    prediction_rows = []
    latest_rows = []
    model_features = {}

    latest_date = features["date"].max()
    latest = features[features["date"] == latest_date].dropna(subset=FEATURES, how="all").copy()
    model_names = model_names or DEFAULT_MODELS
    horizons = horizons or HORIZONS

    for horizon in horizons:
        target = f"future_return_{horizon}d"
        usable = features.dropna(subset=[target]).copy()
        train, validation, test = chronological_split(usable, horizon)
        columns = [column for column in FEATURES if train[column].notna().any()]
        # 因子筛选只看训练期；OLS3取前2个，4个可选模型取前30个。
        ranked = train[columns].corrwith(train[target]).abs().dropna().sort_values(ascending=False)
        candidates_by_name = build_models(model_names, device)

        for model_name in model_names:
            print(f"训练 {model_name} / {horizon} 日 ...", flush=True)

            model_columns = columns
            if model_name == "ols3":
                model_columns = ranked.head(2).index.tolist()
            elif model_name in {"lstm", "transformer", "realmlp", "tabpfn"}:
                model_columns = ranked.head(30).index.tolist()
            model_features[f"{model_name}_{horizon}d"] = model_columns
            train_validation = pd.concat([train, validation], ignore_index=True)
            test_rows = test
            if model_name in {"lstm", "transformer"}:
                _, selected_state = fit_sequence(
                    model_name, train, validation, model_columns, target, nn_epochs, device
                )
                best_model, state = fit_sequence(
                    model_name, train_validation, pd.DataFrame(), model_columns,
                    target, selected_state["epochs"], device,
                )
                test_values, test_ends, test_rows = sequence_data(
                    test, model_columns, state["median"], state["std"]
                )
                test_pred = predict_sequence(best_model, test_values, test_ends, state, device)
                best_label = f"window=20,epochs={selected_state['epochs']}"
                # 输入层权重仅描述网络连接强度，不与树的重要性或回归系数比较。
                values = best_model.projection.weight.detach().abs().mean(dim=0).cpu().numpy()
                importance_type = "输入层平均绝对权重（非预测重要性）"
            else:
                best_label = ""
                best_model = None
                best_mae = np.inf
                fitted_train = fit_rows(train, model_name)
                for label, candidate in candidates_by_name[model_name]:
                    candidate.fit(fitted_train[model_columns], fitted_train[target])
                    validation_pred = predict_model(candidate, validation[model_columns], model_name)
                    validation_mae = mean_absolute_error(validation[target], validation_pred)
                    if validation_mae < best_mae:
                        best_label, best_model, best_mae = label, candidate, validation_mae
                fitted = fit_rows(train_validation, model_name)
                best_model.fit(fitted[model_columns], fitted[target])
                test_pred = predict_model(best_model, test[model_columns], model_name)
                values, importance_type = feature_importance(
                    best_model, test[model_columns], test[target], model_name
                )

            metric_rows.append(
                {
                    "model": model_name,
                    "horizon": horizon,
                    "best_params": best_label,
                    "train_start": train["date"].min(),
                    "train_end": train["date"].max(),
                    "validation_start": validation["date"].min(),
                    "validation_end": validation["date"].max(),
                    "test_start": test["date"].min(),
                    "test_end": test["date"].max(),
                    "n_train": len(train),
                    "n_validation": len(validation),
                    "n_test": len(test_rows),
                    "n_features": len(model_columns),
                    "fit_rows": state["fit_rows"] if model_name in {"lstm", "transformer"} else len(fitted),
                    **regression_metrics(test_rows[target], test_pred),
                }
            )

            predictions = test_rows[["date", "code", "name", target]].copy()
            predictions = predictions.rename(columns={target: "actual_return"})
            predictions["predicted_return"] = test_pred
            predictions["model"] = model_name
            predictions["horizon"] = horizon
            prediction_rows.append(predictions)

            for factor, value in zip(model_columns, values):
                importance_rows.append(
                    {
                        "model": model_name,
                        "horizon": horizon,
                        "factor": factor,
                        "importance": float(value),
                        "importance_type": importance_type,
                    }
                )

            if model_name in {"lstm", "transformer"}:
                best_model, state = fit_sequence(
                    model_name, usable, pd.DataFrame(), model_columns,
                    target, selected_state["epochs"], device,
                )
                state["state_dict"] = {key: value.detach().cpu() for key, value in best_model.state_dict().items()}
                joblib.dump(state, MODEL_DIR / f"{model_name}_{horizon}d.joblib")
                history = features.sort_values("date").groupby("code", sort=False).tail(20)
                latest_values, latest_ends, latest_rows_frame = sequence_data(
                    history, model_columns, state["median"], state["std"]
                )
                latest_pred = predict_sequence(best_model, latest_values, latest_ends, state, device)
                latest_prediction = latest_rows_frame[["date", "code", "name", "close"]].copy()
                latest_prediction["predicted_return"] = latest_pred
                latest_prediction = latest_prediction[latest_prediction["date"] == latest_date].copy()
            else:
                final_train = fit_rows(usable, model_name)
                best_model.fit(final_train[model_columns], final_train[target])
                joblib.dump(best_model, MODEL_DIR / f"{model_name}_{horizon}d.joblib")
                latest_prediction = latest[["date", "code", "name", "close"]].copy()
                latest_prediction["predicted_return"] = predict_model(best_model, latest[model_columns], model_name)
            latest_prediction["reference_target_price"] = (
                latest_prediction["close"] * (1 + latest_prediction["predicted_return"])
            )
            latest_prediction["model"] = model_name
            latest_prediction["horizon"] = horizon
            latest_rows.append(latest_prediction)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)

    metrics.to_csv(RESULT_DIR / "model_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(RESULT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(importance_rows).to_csv(
        RESULT_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(latest_rows, ignore_index=True).to_csv(
        RESULT_DIR / "latest_predictions.csv", index=False, encoding="utf-8-sig"
    )

    metadata = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "latest_market_date": str(pd.Timestamp(latest_date).date()),
        "features": FEATURES,
        "horizons": list(horizons),
        "models": model_names,
        "model_features": model_features,
        "tabpfn_train_limit": 128 if "tabpfn" in model_names else None,
        "stock_count": int(features["code"].nunique()),
    }
    (RESULT_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics, predictions
