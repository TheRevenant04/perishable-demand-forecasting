"""Evaluate the tuned tree model on the untouched test split.

Mirrors ``evaluate_ridge_forecast_test.py`` so the naive baseline, Ridge
and the tuned tree model are compared with the same metric protocol.

Usage:
    uv run python -m perishable_demand_forecasting.forecasting.evaluate_tree_forecast_test
"""

from pathlib import Path
import pickle

import numpy as np
import polars as pl

from perishable_demand_forecasting.config import (
    FORECASTING_DIR,
    FORECASTING_MODELS_DIR,
)

TEST_PATH = FORECASTING_DIR / "test.parquet"
MODEL_PATH = FORECASTING_MODELS_DIR / "tree_forecast_tuned.pkl"
FINAL_MODEL_PATH = FORECASTING_MODELS_DIR / "final_demand_forecast.pkl"
OUTPUT_PATH = FORECASTING_DIR / "tree_test_metrics.csv"

TARGET_COLUMN = "target_daily_demand"
HORIZON_COLUMN = "forecast_horizon"
NAIVE_PREDICTION_COLUMN = "demand_lag_1"


def load_test_dataset(
    path: Path,
    features: list[str],
) -> pl.DataFrame:
    """Load test features, target and baseline prediction."""

    if not path.exists():
        raise FileNotFoundError(f"Test dataset not found: {path}")

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

    test = pl.read_parquet(path, columns=required_columns)

    print(f"Test rows: {test.height:,}")

    return test


def calculate_model_metrics(
    data: pl.DataFrame,
    prediction_column: str,
    model_name: str,
) -> pl.DataFrame:
    """Compute overall and per-horizon metrics for one predictor."""

    evaluated = (
        data.select(
            HORIZON_COLUMN,
            TARGET_COLUMN,
            prediction_column,
        )
        .with_columns(
            (pl.col(prediction_column) - pl.col(TARGET_COLUMN)).alias("error")
        )
        .with_columns(
            pl.col("error").abs().alias("absolute_error"),
            pl.col("error").pow(2).alias("squared_error"),
            (pl.col(prediction_column) < 0).alias("negative_prediction"),
        )
    )

    overall = evaluated.select(
        pl.lit(model_name).alias("model"),
        pl.lit("overall").alias("evaluation_scope"),
        pl.lit(None, dtype=pl.Int64).alias(HORIZON_COLUMN),
        pl.len().alias("observations"),
        pl.col("absolute_error").mean().alias("mae"),
        pl.col("squared_error").mean().sqrt().alias("rmse"),
        pl.col("error").mean().alias("bias"),
        (pl.col("negative_prediction").mean() * 100).alias(
            "negative_prediction_pct"
        ),
    )

    by_horizon = (
        evaluated.group_by(HORIZON_COLUMN)
        .agg(
            pl.len().alias("observations"),
            pl.col("absolute_error").mean().alias("mae"),
            pl.col("squared_error").mean().sqrt().alias("rmse"),
            pl.col("error").mean().alias("bias"),
            (pl.col("negative_prediction").mean() * 100).alias(
                "negative_prediction_pct"
            ),
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

    return pl.concat([overall, by_horizon], how="vertical")


def evaluate_tree_forecast_test() -> None:
    print("=" * 75)
    print("FINAL TEST EVALUATION: TUNED TREE MODEL")
    print("=" * 75)

    model_path = MODEL_PATH
    if FINAL_MODEL_PATH.exists():
        with FINAL_MODEL_PATH.open("rb") as file:
            final_bundle = pickle.load(file)
        if "family" in final_bundle:
            model_path = FINAL_MODEL_PATH

    if not model_path.exists():
        raise FileNotFoundError(
            f"Tuned tree model not found: {model_path}. "
            "Run the tree tuning script first."
        )

    print(f"\nLoading tuned tree model: {model_path}")

    with model_path.open("rb") as file:
        bundle = pickle.load(file)

    model = bundle["model"]
    features = bundle["features"]
    family = bundle["family"]
    params = bundle["params"]

    print(f"Family: {family}")
    print(f"Params: {params}")

    print("\nLoading untouched test dataset...")

    test = load_test_dataset(TEST_PATH, features)

    print("\nValidating test features...")

    X_test = np.asarray(
        test.select(features).to_numpy(),
        dtype=np.float32,
    )

    if not np.isfinite(X_test).all():
        raise ValueError("Test features contain non-finite values.")

    print("\nGenerating tree predictions...")

    predictions = model.predict(X_test)

    if not np.isfinite(predictions).all():
        raise ValueError("Tree model generated non-finite predictions.")

    del X_test

    evaluated_test = test.with_columns(
        pl.Series("tree_prediction", predictions)
    )

    print("\nEvaluating Naive baseline...")
    naive_metrics = calculate_model_metrics(
        evaluated_test,
        NAIVE_PREDICTION_COLUMN,
        "naive",
    )

    print("Evaluating tuned tree model...")
    tree_metrics = calculate_model_metrics(
        evaluated_test,
        "tree_prediction",
        family,
    )

    metrics = pl.concat(
        [naive_metrics, tree_metrics],
        how="vertical",
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_csv(OUTPUT_PATH)

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
        ).to_pandas().to_string(index=False)
    )

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
        ).to_pandas().to_string(index=False)
    )

    overall = metrics.filter(
        pl.col("evaluation_scope") == "overall"
    )

    naive_mae = overall.filter(pl.col("model") == "naive")["mae"][0]
    tree_mae = overall.filter(pl.col("model") == family)["mae"][0]

    print("\n" + "=" * 75)
    print("FINAL MODEL COMPARISON")
    print("=" * 75)

    print(f"Naive MAE:          {naive_mae:.6f}")
    print(f"Tree MAE:           {tree_mae:.6f}")
    print(f"MAE improvement:    {(naive_mae - tree_mae) / naive_mae * 100:.2f}%")

    print(f"\nResults saved to: {OUTPUT_PATH}")

    print("\n" + "=" * 75)
    print("TREE TEST EVALUATION COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    evaluate_tree_forecast_test()
