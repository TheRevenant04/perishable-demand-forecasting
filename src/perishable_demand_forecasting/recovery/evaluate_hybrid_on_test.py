from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TEST_FILE = Path(
    "data/processed/freshretailnet/recovery/test.parquet"
)

MODEL_DIR = Path("models/recovery/regularization")
OUTPUT_DIR = Path("data/processed/freshretailnet/recovery")

MODEL_ALPHA_1 = MODEL_DIR / "G_business_hourly_alpha_1.joblib"
MODEL_ALPHA_10 = MODEL_DIR / "G_business_hourly_alpha_10.joblib"

TARGET = "target_daily_demand"
BASELINE = "simulated_observed_daily_sales"
STOCKOUT_HOURS = "simulated_stockout_hours"

FEATURES = [
    "simulated_observed_daily_sales",
    "simulated_stockout_hours",
    "city_id",
    "store_id",
    "product_id",
    "day_of_week",
    "month",
    "discount",
    "holiday_flag",
    "activity_flag",
    *[
        f"observed_hour_{hour:02d}"
        for hour in range(6, 22)
    ],
]

SEVERITY_GROUPS = [
    ("0 hours", 0, 0),
    ("1-2 hours", 1, 2),
    ("3-5 hours", 3, 5),
    ("6-9 hours", 6, 9),
    ("10-15 hours", 10, 15),
    ("16 hours", 16, 16),
]


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def calculate_metrics(
    actual: np.ndarray,
    predictions: np.ndarray,
) -> dict:
    errors = predictions - actual

    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors**2))),
        "Bias": float(np.mean(errors)),
        "Negative_predictions_pct": float(
            np.mean(predictions < 0) * 100
        ),
    }


def evaluate_predictions(
    actual: np.ndarray,
    predictions: dict[str, np.ndarray],
    stockout_hours: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    overall_rows = []

    for strategy, prediction in predictions.items():
        overall_rows.append({
            "strategy": strategy,
            "rows": len(actual),
            **calculate_metrics(actual, prediction),
        })

    severity_rows = []

    for label, minimum, maximum in SEVERITY_GROUPS:
        mask = (
            (stockout_hours >= minimum)
            & (stockout_hours <= maximum)
        )

        if not np.any(mask):
            continue

        for strategy, prediction in predictions.items():
            severity_rows.append({
                "severity": label,
                "rows": int(mask.sum()),
                "strategy": strategy,
                **calculate_metrics(
                    actual[mask],
                    prediction[mask],
                ),
            })

    return (
        pd.DataFrame(overall_rows),
        pd.DataFrame(severity_rows),
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print("=" * 85)
    print("DAILY DEMAND RECOVERY: FINAL TEST EVALUATION")
    print("=" * 85)

    # -----------------------------------------------------------------
    # Load untouched test dataset
    # -----------------------------------------------------------------

    print("\nLoading untouched test dataset...")

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_FILE}"
        )

    test = pl.read_parquet(TEST_FILE)

    print(f"Test rows: {test.height:,}")

    # -----------------------------------------------------------------
    # Validate source columns
    # -----------------------------------------------------------------

    source_columns = [
        TARGET,
        BASELINE,
        STOCKOUT_HOURS,
        "dt",
        "city_id",
        "store_id",
        "product_id",
        "discount",
        "holiday_flag",
        "activity_flag",
        *[
            f"observed_hour_{hour:02d}"
            for hour in range(6, 22)
        ],
    ]

    missing_columns = [
        column
        for column in source_columns
        if column not in test.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required source columns: "
            + ", ".join(missing_columns)
        )

    # -----------------------------------------------------------------
    # Parse date and derive calendar features
    # -----------------------------------------------------------------

    print("\nPreparing test features...")

    test = test.with_columns(
        pl.col("dt")
        .str.to_date(strict=True)
        .alias("dt")
    )

    test = test.with_columns([
        pl.col("city_id").cast(pl.String),
        pl.col("store_id").cast(pl.String),
        pl.col("product_id").cast(pl.String),
        pl.col("dt")
        .dt.weekday()
        .cast(pl.Int8)
        .alias("day_of_week"),
        pl.col("dt")
        .dt.month()
        .cast(pl.Int8)
        .alias("month"),
    ])

    # -----------------------------------------------------------------
    # Validate final model features
    # -----------------------------------------------------------------

    missing_features = [
        column
        for column in FEATURES
        if column not in test.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing model features after preparation: "
            + ", ".join(missing_features)
        )

    X_test = test.select(FEATURES).to_pandas()

    actual = test.get_column(TARGET).to_numpy()

    observed_sales = test.get_column(
        BASELINE
    ).to_numpy()

    stockout_hours = test.get_column(
        STOCKOUT_HOURS
    ).to_numpy()

    # -----------------------------------------------------------------
    # Load trained models
    # -----------------------------------------------------------------

    print("\nLoading trained Ridge models...")

    for model_path in [
        MODEL_ALPHA_1,
        MODEL_ALPHA_10,
    ]:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

    model_alpha_1 = joblib.load(
        MODEL_ALPHA_1
    )

    model_alpha_10 = joblib.load(
        MODEL_ALPHA_10
    )

    # -----------------------------------------------------------------
    # Generate predictions
    # -----------------------------------------------------------------

    print("\nGenerating predictions...")

    prediction_alpha_1 = model_alpha_1.predict(
        X_test
    )

    prediction_alpha_10 = model_alpha_10.predict(
        X_test
    )

    # -----------------------------------------------------------------
    # Frozen hybrid strategy
    #
    # 1-5 hours  -> observed daily sales
    # 6-9 hours  -> Ridge alpha 10
    # 10-16      -> Ridge alpha 1
    #
    # This routing is for retrospective recovery only.
    # -----------------------------------------------------------------

    hybrid_predictions = np.select(
        [
            stockout_hours <= 5,
            (stockout_hours >= 6)
            & (stockout_hours <= 9),
            stockout_hours >= 10,
        ],
        [
            observed_sales,
            prediction_alpha_10,
            prediction_alpha_1,
        ],
        default=observed_sales,
    )

    predictions = {
        "observed_sales_baseline": observed_sales,
        "ridge_alpha_1": prediction_alpha_1,
        "ridge_alpha_10": prediction_alpha_10,
        "frozen_hybrid": hybrid_predictions,
    }

    # -----------------------------------------------------------------
    # Evaluate predictions
    # -----------------------------------------------------------------

    print("\nEvaluating test predictions...")

    overall, severity = evaluate_predictions(
        actual,
        predictions,
        stockout_hours,
    )

    # -----------------------------------------------------------------
    # Display overall results
    # -----------------------------------------------------------------

    print("\n" + "=" * 85)
    print("OVERALL TEST COMPARISON")
    print("=" * 85)

    print(
        overall.to_string(
            index=False,
            formatters={
                "MAE": "{:.6f}".format,
                "RMSE": "{:.6f}".format,
                "Bias": "{:.6f}".format,
                "Negative_predictions_pct": "{:.2f}%".format,
            },
        )
    )

    # -----------------------------------------------------------------
    # Display severity results
    # -----------------------------------------------------------------

    print("\n" + "=" * 85)
    print("TEST RESULTS BY STOCKOUT SEVERITY")
    print("=" * 85)

    for severity_label in severity["severity"].unique():

        group = severity[
            severity["severity"] == severity_label
        ]

        print(f"\n{severity_label}")

        print(
            group.to_string(
                index=False,
                formatters={
                    "MAE": "{:.6f}".format,
                    "RMSE": "{:.6f}".format,
                    "Bias": "{:.6f}".format,
                    "Negative_predictions_pct": "{:.2f}%".format,
                },
            )
        )

    # -----------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall_path = (
        OUTPUT_DIR / "hybrid_test_overall.csv"
    )

    severity_path = (
        OUTPUT_DIR / "hybrid_test_severity.csv"
    )

    overall.to_csv(
        overall_path,
        index=False,
    )

    severity.to_csv(
        severity_path,
        index=False,
    )

    # -----------------------------------------------------------------
    # Completion
    # -----------------------------------------------------------------

    print("\n" + "=" * 85)
    print("RESULTS SAVED")
    print("=" * 85)

    print(f"Overall: {overall_path}")
    print(f"Severity: {severity_path}")

    print("\n" + "=" * 85)
    print("FINAL TEST EVALUATION COMPLETE")
    print("=" * 85)


if __name__ == "__main__":
    main()