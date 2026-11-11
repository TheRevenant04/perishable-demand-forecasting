
from pathlib import Path

import joblib
import numpy as np
import polars as pl

TEST_PATH = Path(
    "data/processed/freshretailnet/forecasting/test.parquet"
)

MODEL_PATH = Path(
    "models/forecasting/ridge_forecast.joblib"
)

OUTPUT_PATH = Path(
    "data/processed/freshretailnet/forecasting/"
    "forecast_test_metrics.csv"
)

TARGET_COLUMN = "target_daily_demand"
HORIZON_COLUMN = "forecast_horizon"
NAIVE_PREDICTION_COLUMN = "demand_lag_1"


def load_test_dataset(
    path: Path,
    features: list[str],
) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {path}"
        )

    required_columns = list(
        dict.fromkeys(
            features
            + [
                TARGET_COLUMN,
                HORIZON_COLUMN,
                NAIVE_PREDICTION_COLUMN,
            ]
        )
    )

    test = pl.read_parquet(
        path,
        columns=required_columns,
    )

    print(f"Test rows: {test.height:,}")

    return test


def calculate_model_metrics(
    data: pl.DataFrame,
    prediction_column: str,
    model_name: str,
) -> pl.DataFrame:
    evaluated = (
        data.select(
            HORIZON_COLUMN,
            TARGET_COLUMN,
            prediction_column,
        )
        .with_columns(
            (
                pl.col(prediction_column)
                - pl.col(TARGET_COLUMN)
            ).alias("error")
        )
        .with_columns(
            pl.col("error").abs().alias("absolute_error"),
            pl.col("error").pow(2).alias("squared_error"),
            (pl.col(prediction_column) < 0).alias(
                "negative_prediction"
            ),
        )
    )

    overall = evaluated.select(
        pl.lit(model_name).alias("model"),
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

    by_horizon = (
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
            pl.lit(model_name).alias("model"),
            pl.lit("by_horizon").alias("evaluation_scope"),
        )
        .select(
            "model",
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

    return pl.concat(
        [overall, by_horizon],
        how="vertical",
    )


def evaluate_ridge_forecast_test() -> None:
    print("=" * 75)
    print("FINAL TEST EVALUATION: NAIVE VS RIDGE REGRESSION")
    print("=" * 75)

    # ---------------------------------------------------------
    # Load trained Ridge model
    # ---------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained Ridge model not found: {MODEL_PATH}"
        )

    print(f"\nLoading Ridge model: {MODEL_PATH}")

    saved_model = joblib.load(MODEL_PATH)

    model = saved_model["model"]
    scaler = saved_model["scaler"]
    features = saved_model["features"]
    alpha = saved_model["alpha"]

    if alpha != 1.0:
        raise ValueError(
            f"Expected Ridge alpha=1.0, found alpha={alpha}."
        )

    print(f"Ridge alpha: {alpha}")

    # ---------------------------------------------------------
    # Load untouched test dataset
    # ---------------------------------------------------------

    print("\nLoading untouched test dataset...")

    test = load_test_dataset(
        TEST_PATH,
        features,
    )

    # ---------------------------------------------------------
    # Validate input data
    # ---------------------------------------------------------

    print("\nValidating test features...")

    X_test = np.asarray(
        test.select(features).to_numpy(),
        dtype=np.float32,
    )

    if not np.isfinite(X_test).all():
        raise ValueError(
            "Test features contain non-finite values."
        )

    if not np.isfinite(
        test[TARGET_COLUMN].to_numpy()
    ).all():
        raise ValueError(
            "Test target contains non-finite values."
        )

    if not np.isfinite(
        test[NAIVE_PREDICTION_COLUMN].to_numpy()
    ).all():
        raise ValueError(
            "Naive predictions contain non-finite values."
        )

    # ---------------------------------------------------------
    # Generate Ridge predictions
    # ---------------------------------------------------------

    print("\nPreparing Ridge test features...")

    X_test = scaler.transform(X_test)

    print("\nGenerating Ridge predictions...")

    ridge_predictions = model.predict(X_test)

    if not np.isfinite(ridge_predictions).all():
        raise ValueError(
            "Ridge generated non-finite predictions."
        )

    print("Ridge predictions generated successfully.")

    del X_test

    # ---------------------------------------------------------
    # Add predictions to test dataset
    # ---------------------------------------------------------

    evaluated_test = test.with_columns(
        pl.Series(
            "ridge_prediction",
            ridge_predictions,
        )
    )

    # ---------------------------------------------------------
    # Evaluate Naive baseline
    # ---------------------------------------------------------

    print("\nEvaluating Naive baseline...")

    naive_metrics = calculate_model_metrics(
        evaluated_test,
        NAIVE_PREDICTION_COLUMN,
        "naive",
    )

    # ---------------------------------------------------------
    # Evaluate Ridge Regression
    # ---------------------------------------------------------

    print("\nEvaluating Ridge Regression...")

    ridge_metrics = calculate_model_metrics(
        evaluated_test,
        "ridge_prediction",
        "ridge_alpha_1",
    )

    # ---------------------------------------------------------
    # Combine and save results
    # ---------------------------------------------------------

    metrics = pl.concat(
        [
            naive_metrics,
            ridge_metrics,
        ],
        how="vertical",
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics.write_csv(OUTPUT_PATH)

    # ---------------------------------------------------------
    # Display overall results
    # ---------------------------------------------------------

    print("\n" + "=" * 75)
    print("OVERALL TEST RESULTS")
    print("=" * 75)

    print(
        metrics.filter(
            pl.col("evaluation_scope") == "overall"
        ).select(
            "model",
            "observations",
            "mae",
            "rmse",
            "bias",
            "negative_prediction_pct",
        )
    )

    # ---------------------------------------------------------
    # Display results by horizon
    # ---------------------------------------------------------

    print("\n" + "=" * 75)
    print("TEST RESULTS BY FORECAST HORIZON")
    print("=" * 75)

    print(
        metrics.filter(
            pl.col("evaluation_scope") == "by_horizon"
        ).select(
            "model",
            HORIZON_COLUMN,
            "observations",
            "mae",
            "rmse",
            "bias",
            "negative_prediction_pct",
        ).sort(
            ["forecast_horizon", "model"]
        )
    )

    # ---------------------------------------------------------
    # Calculate MAE improvement
    # ---------------------------------------------------------

    overall = metrics.filter(
        pl.col("evaluation_scope") == "overall"
    )

    naive_mae = overall.filter(
        pl.col("model") == "naive"
    )["mae"][0]

    ridge_mae = overall.filter(
        pl.col("model") == "ridge_alpha_1"
    )["mae"][0]

    improvement = (
        (naive_mae - ridge_mae)
        / naive_mae
    ) * 100

    print("\n" + "=" * 75)
    print("FINAL MODEL COMPARISON")
    print("=" * 75)

    print(f"Naive MAE:          {naive_mae:.6f}")
    print(f"Ridge MAE:          {ridge_mae:.6f}")
    print(f"MAE improvement:    {improvement:.2f}%")

    print(f"\nResults saved to: {OUTPUT_PATH}")

    print("\n" + "=" * 75)
    print("FINAL TEST EVALUATION COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    evaluate_ridge_forecast_test()
