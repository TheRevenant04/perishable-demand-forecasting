
from pathlib import Path

import polars as pl

DATA_PATH = Path(
    "data/processed/freshretailnet/forecasting/validation.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/freshretailnet/forecasting/"
    "naive_validation_metrics.csv"
)

PREDICTION_COLUMN = "demand_lag_1"
TARGET_COLUMN = "target_daily_demand"
HORIZON_COLUMN = "forecast_horizon"


def evaluate_naive_forecast() -> None:
    print("=" * 75)
    print("NAIVE FORECAST: VALIDATION EVALUATION")
    print("=" * 75)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Validation dataset not found: {DATA_PATH}"
        )

    print(f"\nLoading validation dataset: {DATA_PATH}")

    data = pl.scan_parquet(DATA_PATH)

    available_columns = data.collect_schema().names()

    required_columns = [
        PREDICTION_COLUMN,
        TARGET_COLUMN,
        HORIZON_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in available_columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Calculate prediction errors.
    evaluated = data.with_columns(
        (
            pl.col(PREDICTION_COLUMN) - pl.col(TARGET_COLUMN)
        ).alias("error")
    ).with_columns(
        pl.col("error").abs().alias("absolute_error"),
        pl.col("error").pow(2).alias("squared_error"),
    )

    # Overall validation metrics.
    overall = evaluated.select(
        pl.lit("overall").alias("evaluation_scope"),
        pl.lit(None, dtype=pl.Int64).alias(HORIZON_COLUMN),
        pl.len().alias("observations"),
        pl.col("absolute_error").mean().alias("mae"),
        pl.col("squared_error").mean().sqrt().alias("rmse"),
        pl.col("error").mean().alias("bias"),
    )

    # Validation metrics for each forecast horizon.
    by_horizon = (
        evaluated
        .group_by(HORIZON_COLUMN)
        .agg(
            pl.len().alias("observations"),
            pl.col("absolute_error").mean().alias("mae"),
            pl.col("squared_error").mean().sqrt().alias("rmse"),
            pl.col("error").mean().alias("bias"),
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
        )
        .sort(HORIZON_COLUMN)
    )

    # Concatenate results after ensuring matching schemas.
    metrics = pl.concat(
        [overall, by_horizon],
        how="vertical",
    ).collect()

    # Save evaluation results.
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics.write_csv(OUTPUT_PATH)

    # Display overall results.
    print("\n" + "=" * 75)
    print("OVERALL NAIVE FORECAST RESULTS")
    print("=" * 75)

    print(
        metrics.filter(
            pl.col("evaluation_scope") == "overall"
        )
    )

    # Display results by forecast horizon.
    print("\n" + "=" * 75)
    print("NAIVE FORECAST RESULTS BY HORIZON")
    print("=" * 75)

    print(
        metrics.filter(
            pl.col("evaluation_scope") == "by_horizon"
        )
    )

    print(f"\nResults saved to: {OUTPUT_PATH}")

    print("\n" + "=" * 75)
    print("NAIVE FORECAST EVALUATION COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    evaluate_naive_forecast()
