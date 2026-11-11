"""Tune tree-based models for seven-day demand forecasting.

Implements the model family taught in ML Zoomcamp module 6
(decision trees, random forest and XGBoost) and tunes their main
regularization parameters on a deterministic sample of the training
split. Every candidate is scored on the full validation split using the
same MAE/RMSE/bias protocol as the Ridge experiments.

Usage:
    uv run python -m perishable_demand_forecasting.forecasting.tune_tree_forecast
"""

from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import pickle
import time

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from perishable_demand_forecasting.config import (
    FORECASTING_DIR,
    FORECASTING_MODELS_DIR,
)

TRAIN_PATH = FORECASTING_DIR / "train.parquet"
VALIDATION_PATH = FORECASTING_DIR / "validation.parquet"
METRICS_PATH = FORECASTING_DIR / "tree_validation_metrics.csv"
BEST_MODEL_PATH = FORECASTING_MODELS_DIR / "tree_forecast_tuned.pkl"

TARGET_COLUMN = "target_daily_demand"
HORIZON_COLUMN = "forecast_horizon"

HISTORICAL_FEATURES = (
    [f"demand_lag_{i}" for i in range(1, 15)]
    + [
        "demand_rolling_mean_7",
        "demand_rolling_mean_14",
        "stockout_hours_lag_1",
        "stockout_hours_lag_7",
    ]
)

FUTURE_FEATURES = [
    HORIZON_COLUMN,
    "target_weekday",
    "target_month",
    "target_holiday_flag",
]

FEATURES = HISTORICAL_FEATURES + FUTURE_FEATURES

# Tuning on the full 12.95M-row training split would take hours for the
# forest models. A deterministic sample keeps the comparison reproducible
# while covering enough rows for stable validation scores.
TREE_SAMPLE_ROWS = 2_000_000
FOREST_SAMPLE_ROWS = 1_000_000
RANDOM_SEED = 42


def load_split(
    path: Path,
    n_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    """Load features and target, optionally sampling rows deterministically."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    frame = pl.read_parquet(path, columns=FEATURES + [TARGET_COLUMN])

    if n_rows is not None and frame.height > n_rows:
        frame = frame.sample(n=n_rows, seed=RANDOM_SEED)

    features = np.asarray(
        frame.select(FEATURES).to_numpy(),
        dtype=np.float32,
    )
    target = np.asarray(
        frame[TARGET_COLUMN].to_numpy(),
        dtype=np.float32,
    )

    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite features in {path}.")

    if not np.isfinite(target).all():
        raise ValueError(f"Non-finite target in {path}.")

    print(f"  {path.name}: {frame.height:,} rows")

    return features, target, frame


def evaluate_predictions(
    frame: pl.DataFrame,
    predictions: np.ndarray,
) -> dict:
    """Compute the validation metrics used across forecasting experiments."""

    evaluated = (
        frame.select(TARGET_COLUMN)
        .with_columns(pl.Series("prediction", predictions))
        .with_columns(
            (pl.col("prediction") - pl.col(TARGET_COLUMN)).alias("error")
        )
    )

    return evaluated.select(
        pl.col("error").abs().mean().alias("mae"),
        pl.col("error").pow(2).mean().sqrt().alias("rmse"),
        pl.col("error").mean().alias("bias"),
        (pl.col("prediction") < 0).mean().mul(100).alias(
            "negative_prediction_pct"
        ),
    ).row(0, named=True)


def fit_and_score(
    model_name: str,
    model,
    params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    validation: pl.DataFrame,
    extra_fit_kwargs: dict | None = None,
) -> tuple[dict, object]:
    """Fit one candidate, score it on validation and return the metrics."""

    print(f"\n  [{model_name}] {params}")

    started = time.perf_counter()

    model.fit(X_train, y_train, **(extra_fit_kwargs or {}))

    predictions = model.predict(X_validation)

    if not np.isfinite(predictions).all():
        raise ValueError(f"{model_name} generated non-finite predictions.")

    metrics = evaluate_predictions(validation, predictions)

    elapsed = time.perf_counter() - started

    print(
        f"    MAE={metrics['mae']:.6f}  RMSE={metrics['rmse']:.6f}  "
        f"bias={metrics['bias']:.6f}  ({elapsed:.1f}s)"
    )

    result = {
        "model": model_name,
        "params": params,
        "fit_seconds": elapsed,
        **metrics,
    }

    return result, model


def tune_decision_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    validation: pl.DataFrame,
) -> tuple[list[dict], dict]:
    """Tune tree depth and minimum leaf size (Zoomcamp 6.5)."""

    results: list[dict] = []
    best: dict | None = None

    for max_depth, min_samples_leaf in product([8, 12, 16], [20, 100]):
        params = {
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
        }

        result, model = fit_and_score(
            "decision_tree",
            DecisionTreeRegressor(random_state=RANDOM_SEED, **params),
            params,
            X_train,
            y_train,
            X_validation,
            validation,
        )
        results.append(result)

        if best is None or result["mae"] < best["mae"]:
            best = {**result, "estimator": model}

    return results, best


def tune_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    validation: pl.DataFrame,
) -> tuple[list[dict], dict]:
    """Tune forest depth and minimum leaf size (Zoomcamp 6.6)."""

    results: list[dict] = []
    best: dict | None = None

    for max_depth, min_samples_leaf in product([10, 14], [20, 100]):
        params = {
            "n_estimators": 60,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_samples": 0.7,
            "max_features": 0.5,
            "n_jobs": -1,
            "random_state": RANDOM_SEED,
        }

        result, model = fit_and_score(
            "random_forest",
            RandomForestRegressor(**params),
            params,
            X_train,
            y_train,
            X_validation,
            validation,
        )
        results.append(result)

        if best is None or result["mae"] < best["mae"]:
            best = {**result, "estimator": model}

    return results, best


def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    validation: pl.DataFrame,
) -> tuple[list[dict], dict]:
    """Tune learning rate and depth with early stopping (Zoomcamp 6.8)."""

    results: list[dict] = []
    best: dict | None = None

    for eta, max_depth in product([0.05, 0.1], [6, 10]):
        params = {
            "n_estimators": 600,
            "learning_rate": eta,
            "max_depth": max_depth,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "early_stopping_rounds": 30,
            "eval_metric": "mae",
            "n_jobs": -1,
            "random_state": RANDOM_SEED,
        }

        model = XGBRegressor(**params)

        result, fitted = fit_and_score(
            "xgboost",
            model,
            {k: v for k, v in params.items() if k != "n_estimators"},
            X_train,
            y_train,
            X_validation,
            validation,
            extra_fit_kwargs={
                "eval_set": [(X_validation, validation[TARGET_COLUMN].to_numpy())],
                "verbose": False,
            },
        )

        result["params"]["n_estimators_fitted"] = int(
            fitted.best_iteration + 1
        )
        results.append(result)

        if best is None or result["mae"] < best["mae"]:
            best = {**result, "estimator": fitted}

    return results, best


def main() -> None:
    print("=" * 75)
    print("TREE-BASED FORECASTING: PARAMETER TUNING")
    print("=" * 75)

    print("\nLoading validation split...")
    X_validation, _, validation = load_split(VALIDATION_PATH)

    print("\nLoading training sample...")
    X_tree, y_tree, _ = load_split(TRAIN_PATH, n_rows=TREE_SAMPLE_ROWS)

    print("\nLoading smaller forest training sample...")
    X_forest, y_forest, _ = load_split(
        TRAIN_PATH,
        n_rows=FOREST_SAMPLE_ROWS,
    )

    all_results: list[dict] = []
    best_by_family: dict[str, dict] = {}

    print("\n" + "-" * 75)
    print("DECISION TREE TUNING")
    print("-" * 75)

    results, best = tune_decision_tree(
        X_tree,
        y_tree,
        X_validation,
        validation,
    )
    all_results.extend(results)
    best_by_family["decision_tree"] = best

    print("\n" + "-" * 75)
    print("RANDOM FOREST TUNING")
    print("-" * 75)

    results, best = tune_random_forest(
        X_forest,
        y_forest,
        X_validation,
        validation,
    )
    all_results.extend(results)
    best_by_family["random_forest"] = best

    print("\n" + "-" * 75)
    print("XGBOOST TUNING")
    print("-" * 75)

    results, best = tune_xgboost(
        X_tree,
        y_tree,
        X_validation,
        validation,
    )
    all_results.extend(results)
    best_by_family["xgboost"] = best

    metrics_frame = (
        pl.DataFrame(
            [
                {
                    "model": result["model"],
                    "params": str(result["params"]),
                    "fit_seconds": result["fit_seconds"],
                    "mae": result["mae"],
                    "rmse": result["rmse"],
                    "bias": result["bias"],
                    "negative_prediction_pct": result[
                        "negative_prediction_pct"
                    ],
                }
                for result in all_results
            ]
        )
        .sort("mae")
    )

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_frame.write_csv(METRICS_PATH)

    best_family = min(
        best_by_family,
        key=lambda family: best_by_family[family]["mae"],
    )
    best_result = best_by_family[best_family]

    FORECASTING_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with BEST_MODEL_PATH.open("wb") as file:
        pickle.dump(
            {
                "model": best_result["estimator"],
                "features": FEATURES,
                "target": TARGET_COLUMN,
                "family": best_family,
                "params": best_result["params"],
                "validation_mae": best_result["mae"],
                "trained_at": datetime.now(timezone.utc).isoformat(),
            },
            file,
        )

    print("\n" + "=" * 75)
    print("TREE MODEL VALIDATION COMPARISON")
    print("=" * 75)
    print(metrics_frame.to_pandas().to_string(index=False))

    print("\n" + "=" * 75)
    print("BEST TREE MODEL PER FAMILY")
    print("=" * 75)

    for family, result in best_by_family.items():
        print(
            f"{family:16s} MAE={result['mae']:.6f}  "
            f"params={result['params']}"
        )

    print(f"\nBest overall family: {best_family}")
    print(f"Validation MAE:      {best_result['mae']:.6f}")
    print(f"\nMetrics saved to: {METRICS_PATH}")
    print(f"Best model saved to: {BEST_MODEL_PATH}")

    print("\n" + "=" * 75)
    print("TREE MODEL TUNING COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()
