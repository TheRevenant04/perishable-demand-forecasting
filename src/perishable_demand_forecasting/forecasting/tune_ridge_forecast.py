
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

TRAIN_PATH = Path(
    "data/processed/freshretailnet/forecasting/train.parquet"
)

VALIDATION_PATH = Path(
    "data/processed/freshretailnet/forecasting/validation.parquet"
)

METRICS_PATH = Path(
    "data/processed/freshretailnet/forecasting/"
    "ridge_alpha_metrics.csv"
)

BEST_MODEL_PATH = Path(
    "models/forecasting/ridge_forecast_tuned.joblib"
)

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

ALPHA_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]


def load_dataset(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    return pl.read_parquet(
        path,
        columns=FEATURES + [TARGET_COLUMN],
    )


def evaluate_predictions(
    data: pl.DataFrame,
    predictions: np.ndarray,
) -> dict:
    evaluated = (
        data.select(
            HORIZON_COLUMN,
            TARGET_COLUMN,
        )
        .with_columns(
            pl.Series("prediction", predictions)
        )
        .with_columns(
            (
                pl.col("prediction")
                - pl.col(TARGET_COLUMN)
            ).alias("error")
        )
    )

    result = evaluated.select(
        pl.col("error").abs().mean().alias("mae"),
        pl.col("error").pow(2).mean().sqrt().alias("rmse"),
        pl.col("error").mean().alias("bias"),
        (
            (pl.col("prediction") < 0).mean() * 100
        ).alias("negative_prediction_pct"),
    ).row(0, named=True)

    return result


def tune_ridge_forecast() -> None:
    print("=" * 75)
    print("RIDGE REGRESSION: REGULARIZATION TUNING")
    print("=" * 75)

    # ---------------------------------------------------------
    # Load training data
    # ---------------------------------------------------------

    print("\nLoading training dataset...")

    train = load_dataset(TRAIN_PATH)

    print(f"Training rows: {train.height:,}")

    X_train = np.array(
        train.select(FEATURES).to_numpy(),
        dtype=np.float32,
        copy=True,
    )

    y_train = np.array(
        train[TARGET_COLUMN].to_numpy(),
        dtype=np.float32,
        copy=True,
    )

    del train

    if not np.isfinite(X_train).all():
        raise ValueError(
            "Training features contain non-finite values."
        )

    if not np.isfinite(y_train).all():
        raise ValueError(
            "Training target contains non-finite values."
        )

    # ---------------------------------------------------------
    # Load validation data
    # ---------------------------------------------------------

    print("\nLoading validation dataset...")

    validation = load_dataset(VALIDATION_PATH)

    print(f"Validation rows: {validation.height:,}")

    X_validation = np.array(
        validation.select(FEATURES).to_numpy(),
        dtype=np.float32,
        copy=True,
    )

    if not np.isfinite(X_validation).all():
        raise ValueError(
            "Validation features contain non-finite values."
        )

    # ---------------------------------------------------------
    # Standardize features
    # ---------------------------------------------------------

    print("\nStandardizing features...")

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_validation = scaler.transform(X_validation)

    # ---------------------------------------------------------
    # Train and evaluate each alpha
    # ---------------------------------------------------------

    results = []
    best_model = None
    best_alpha = None
    best_mae = float("inf")

    for alpha in ALPHA_VALUES:
        print("\n" + "-" * 75)
        print(f"Training Ridge Regression with alpha={alpha}")
        print("-" * 75)

        model = Ridge(
            alpha=alpha,
            solver="lsqr",
            copy_X=False,
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_validation)

        if not np.isfinite(predictions).all():
            raise ValueError(
                f"Non-finite predictions generated for alpha={alpha}."
            )

        metrics = evaluate_predictions(
            validation,
            predictions,
        )

        result = {
            "alpha": alpha,
            **metrics,
        }

        results.append(result)

        print(f"MAE:                     {metrics['mae']:.6f}")
        print(f"RMSE:                    {metrics['rmse']:.6f}")
        print(f"Bias:                    {metrics['bias']:.6f}")
        print(
            "Negative predictions:    "
            f"{metrics['negative_prediction_pct']:.4f}%"
        )

        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_alpha = alpha
            best_model = model

    # ---------------------------------------------------------
    # Save tuning metrics
    # ---------------------------------------------------------

    metrics_df = (
        pl.DataFrame(results)
        .sort("mae")
    )

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_df.write_csv(METRICS_PATH)

    # ---------------------------------------------------------
    # Save best model separately
    # ---------------------------------------------------------

    BEST_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": best_model,
            "scaler": scaler,
            "features": FEATURES,
            "target": TARGET_COLUMN,
            "alpha": best_alpha,
        },
        BEST_MODEL_PATH,
    )

    # ---------------------------------------------------------
    # Display results
    # ---------------------------------------------------------

    print("\n" + "=" * 75)
    print("RIDGE REGULARIZATION COMPARISON")
    print("=" * 75)

    print(metrics_df)

    print("\n" + "=" * 75)
    print("BEST VALIDATION MODEL")
    print("=" * 75)

    print(f"Best alpha: {best_alpha}")
    print(f"Best validation MAE: {best_mae:.6f}")

    print(f"\nMetrics saved to: {METRICS_PATH}")
    print(f"Best model saved to: {BEST_MODEL_PATH}")

    print("\n" + "=" * 75)
    print("RIDGE REGULARIZATION TUNING COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    tune_ridge_forecast()
