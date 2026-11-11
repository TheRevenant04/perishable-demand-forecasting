
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

MODEL_PATH = Path(
    "models/forecasting/ridge_forecast.joblib"
)

METRICS_PATH = Path(
    "data/processed/freshretailnet/forecasting/"
    "ridge_validation_metrics.csv"
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

RIDGE_ALPHA = 1.0


def load_dataset(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    return pl.read_parquet(
        path,
        columns=FEATURES + [TARGET_COLUMN],
    )


def calculate_metrics(
    data: pl.DataFrame,
    predictions: np.ndarray,
    scope: str,
) -> pl.DataFrame:

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
        .with_columns(
            pl.col("error").abs().alias("absolute_error"),
            pl.col("error").pow(2).alias("squared_error"),
            (pl.col("prediction") < 0).alias(
                "negative_prediction"
            ),
        )
    )

    if scope == "overall":
        return evaluated.select(
            pl.lit("overall").alias("evaluation_scope"),
            pl.lit(None, dtype=pl.Int64).alias(
                HORIZON_COLUMN
            ),
            pl.len().alias("observations"),
            pl.col("absolute_error").mean().alias("mae"),
            pl.col("squared_error").mean().sqrt().alias("rmse"),
            pl.col("error").mean().alias("bias"),
            (
                pl.col("negative_prediction").mean() * 100
            ).alias("negative_prediction_pct"),
        )

    return (
        evaluated
        .group_by(HORIZON_COLUMN)
        .agg(
            pl.len().alias("observations"),
            pl.col("absolute_error").mean().alias("mae"),
            pl.col("squared_error").mean().sqrt().alias("rmse"),
            pl.col("error").mean().alias("bias"),
            (
                pl.col("negative_prediction").mean() * 100
            ).alias("negative_prediction_pct"),
        )
        .with_columns(
            pl.col(HORIZON_COLUMN).cast(pl.Int64),
            pl.lit("by_horizon").alias("evaluation_scope"),
        )
        .select(
            "evaluation_scope",
            HORIZON_COLUMN,
            "observations",
            "mae",
            "rmse",
            "bias",
            "negative_prediction_pct",
        )
        .sort(HORIZON_COLUMN)
    )


def train_ridge_forecast() -> None:
    print("=" * 75)
    print("RIDGE REGRESSION: 7-DAY DEMAND FORECASTING")
    print("=" * 75)

    print("\nForecast features:")
    for feature in FEATURES:
        print(f"  - {feature}")

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
    # Standardize features
    # ---------------------------------------------------------

    print("\nStandardizing features...")

    scaler = StandardScaler(copy=False)
    X_train = scaler.fit_transform(X_train)

    # ---------------------------------------------------------
    # Train Ridge Regression
    # ---------------------------------------------------------

    print(
        f"\nTraining Ridge Regression "
        f"(alpha={RIDGE_ALPHA})..."
    )

    model = Ridge(
        alpha=RIDGE_ALPHA,
        solver="lsqr",
        copy_X=False,
    )

    model.fit(X_train, y_train)

    print("Ridge Regression training complete.")

    del X_train
    del y_train

    # ---------------------------------------------------------
    # Save model and scaler
    # ---------------------------------------------------------

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "features": FEATURES,
            "target": TARGET_COLUMN,
            "alpha": RIDGE_ALPHA,
        },
        MODEL_PATH,
    )

    print(f"\nModel saved to: {MODEL_PATH}")

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
    # Generate predictions
    # ---------------------------------------------------------

    print("\nGenerating validation predictions...")

    X_validation = scaler.transform(X_validation)

    predictions = model.predict(X_validation)

    if not np.isfinite(predictions).all():
        raise ValueError(
            "Model generated non-finite predictions."
        )

    # ---------------------------------------------------------
    # Evaluate predictions
    # ---------------------------------------------------------

    print("\nEvaluating validation predictions...")

    overall = calculate_metrics(
        validation,
        predictions,
        "overall",
    )

    by_horizon = calculate_metrics(
        validation,
        predictions,
        "by_horizon",
    )

    metrics = pl.concat(
        [overall, by_horizon],
        how="vertical",
    )

    # ---------------------------------------------------------
    # Save metrics
    # ---------------------------------------------------------

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics.write_csv(METRICS_PATH)

    # ---------------------------------------------------------
    # Display results
    # ---------------------------------------------------------

    print("\n" + "=" * 75)
    print("OVERALL RIDGE FORECAST RESULTS")
    print("=" * 75)

    print(overall)

    print("\n" + "=" * 75)
    print("RIDGE FORECAST RESULTS BY HORIZON")
    print("=" * 75)

    print(by_horizon)

    print(f"\nMetrics saved to: {METRICS_PATH}")

    print("\n" + "=" * 75)
    print("RIDGE FORECAST TRAINING COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    train_ridge_forecast()
