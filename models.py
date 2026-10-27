"""模型定义、时间切分、训练、测试评估与最新预测。"""

import json

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

from price_source import MODEL_DIR, RESULT_DIR, PRICE_FILE, prepare_folders, source_signature
from factors import FEATURES, HORIZONS
from factor_selection import (
    PREPROCESSING_VERSION, FactorScreen, selection_configs, make_preprocessor,
    preprocessing_summary, pca_loadings, choose_validation_trial,
)


RANDOM_STATE = 42


def linear_pipeline(model) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def build_models() -> dict[str, list[tuple[str, object]]]:
    """基础估计器参数候选；正式train_models会先接入筛选/降维预处理。"""
    return {
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
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
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
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "direction_accuracy": float(np.mean(np.sign(y_true.to_numpy()) == np.sign(y_pred))),
        "prediction_correlation": float(pd.Series(y_true.to_numpy()).corr(pd.Series(y_pred))),
    }


def feature_importance(model: Pipeline) -> np.ndarray:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "coef_"):
        values = np.asarray(estimator.coef_, dtype=float)
    else:
        values = np.asarray(estimator.feature_importances_, dtype=float)
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


def tune_models(train, validation, target, screen, counts=(20, 40, 60, 80),
                pca_variances=(.90, .95), relative_tolerance=0.):
    """预处理每个候选只拟合一次，然后供所有模型参数复用；测试集不是本函数参数。"""
    families = build_models()
    all_trials = []
    for config in selection_configs(screen, counts, pca_variances):
        prep = make_preprocessor(screen, config)
        x_train = prep.fit_transform(train[FEATURES])
        x_valid = prep.transform(validation[FEATURES])
        details = preprocessing_summary(prep, screen, config)
        print(f"  验证 {config.name}：{details['selected_factor_count']}因子 → {details['model_input_dimensions']}维", flush=True)
        for model_name, candidates in families.items():
            for label, pipeline in candidates:
                estimator = clone(pipeline.named_steps["model"])
                estimator.fit(x_train, train[target])
                prediction = estimator.predict(x_valid)
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


def train_models(features: pd.DataFrame, relative_tolerance: float = 0.) -> tuple[pd.DataFrame, pd.DataFrame]:
    """训练期筛选，验证期选择数量/PCA/参数，测试期只评估，成熟全样本另拟合生产模型。"""
    prepare_folders()
    if not 0 <= relative_tolerance <= .1:
        raise ValueError("验证MAE相对容忍度必须位于0至0.1")
    metric_rows = []
    importance_rows = []
    prediction_rows = []
    latest_rows = []
    trial_rows, selection_rows, preprocessing_rows, loading_rows = [], [], [], []
    production_inputs = {}

    latest_date = features["date"].max()
    latest = features[features["date"] == latest_date].dropna(subset=FEATURES, how="all").copy()

    for horizon in HORIZONS:
        target = f"future_return_{horizon}d"
        usable = horizon_samples(features, horizon)
        train, validation, test = chronological_split(usable, horizon)
        print(f"筛选与验证 / {horizon}日 ...", flush=True)
        screen = FactorScreen().fit(train[FEATURES], train[target], train["date"])
        winners, trials = tune_models(train, validation, target, screen, relative_tolerance=relative_tolerance)
        for row in trials:
            trial_rows.append({"horizon": horizon, **{k: v for k, v in row.items() if k not in ("config", "estimator")},
                               "chosen": winners[row["model"]]["trial_id"] == row["trial_id"]})
        train_validation = pd.concat([train, validation], ignore_index=True)
        evaluation_screen = FactorScreen().fit(train_validation[FEATURES], train_validation[target], train_validation["date"])

        for model_name, winner in winners.items():
            config = winner["config"]
            print(f"训练 {model_name} / {horizon} 日：{config.name} / {winner['params']}", flush=True)
            best_model = fit_selected_model(train_validation, target, evaluation_screen, config, winner["estimator"])
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
        production_screen = FactorScreen().fit(usable[FEATURES], usable[target], usable["date"])
        for model_name, winner in winners.items():
            config = winner["config"]
            production_model = fit_selected_model(usable, target, production_screen, config, winner["estimator"])
            joblib.dump(production_model, MODEL_DIR / f"{model_name}_{horizon}d.joblib")
            prep = production_model.named_steps["preprocess"]
            summary = preprocessing_summary(prep, production_screen, config)
            production_inputs[f"{model_name}_{horizon}d"] = {
                "selected_factors": production_screen.selected_features(config), **summary,
            }
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

    metrics.to_csv(RESULT_DIR / "model_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(RESULT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(importance_rows).to_csv(
        RESULT_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(latest_rows, ignore_index=True).to_csv(
        RESULT_DIR / "latest_predictions.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(trial_rows).to_csv(RESULT_DIR / "model_selection_trials.csv", index=False, encoding="utf-8-sig")
    pd.concat(selection_rows, ignore_index=True).to_csv(RESULT_DIR / "factor_selection.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(preprocessing_rows).to_csv(RESULT_DIR / "preprocessing_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(loading_rows, ignore_index=True).to_csv(RESULT_DIR / "pca_loadings.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "data_source": PRICE_FILE.name,
        "preprocessing_version": PREPROCESSING_VERSION,
        "model_inputs": production_inputs,
        "source_sha256": source_signature(),
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "latest_market_date": str(pd.Timestamp(latest_date).date()),
        "features": FEATURES,
        "horizons": list(HORIZONS),
        "stock_count": int(features["code"].nunique()),
    }
    (RESULT_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return metrics, predictions


def audit_factor_selection(features: pd.DataFrame):
    """只检查各预测周期的训练期筛选和候选维数；不选择验证赢家，不生成预测。"""
    folder = RESULT_DIR / "selection_audit"
    folder.mkdir(parents=True, exist_ok=True)
    audits, summaries = [], []
    for horizon in HORIZONS:
        target = f"future_return_{horizon}d"
        train, _, _ = chronological_split(horizon_samples(features, horizon), horizon)
        screen = FactorScreen().fit(train[FEATURES], train[target], train["date"])
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
