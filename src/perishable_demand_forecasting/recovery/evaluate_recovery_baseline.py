
from pathlib import Path

import numpy as np
import polars as pl

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RECOVERY_DIR = Path("data/processed/freshretailnet/recovery")

TRAIN_FILE = RECOVERY_DIR / "train.parquet"
VALIDATION_FILE = RECOVERY_DIR / "validation.parquet"

OUTPUT_FILE = RECOVERY_DIR / "baseline_metrics.csv"

TARGET_COLUMN = "target_daily_demand"
OBSERVED_COLUMN = "simulated_observed_daily_sales"
STOCKOUT_HOURS_COLUMN = "simulated_stockout_hours"

STOCKOUT_GROUPS = [
    ("Fully in stock", 0, 0),
    ("Low", 1, 2),
    ("Moderate", 3, 5),
    ("High", 6, 9),
    ("Very high", 10, 15),
    ("Fully out of stock", 16, 16),
]


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_dataset(path: Path, split_name: str) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{split_name} dataset not found: {path}"
        )

    print(f"Loading {split_name} dataset: {path}")

    df = pl.read_parquet(path)

    required_columns = [
        TARGET_COLUMN,
        OBSERVED_COLUMN,
        STOCKOUT_HOURS_COLUMN,
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{split_name} dataset is missing columns: {missing}"
        )

    if df.is_empty():
        raise ValueError(
            f"{split_name} dataset is empty."
        )

    return df


# ---------------------------------------------------------------------
# Metric calculation
# ---------------------------------------------------------------------

def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict:

    if len(actual) == 0:
        raise ValueError("Cannot calculate metrics for an empty group.")

    errors = predicted - actual

    mae = np.mean(np.abs(errors))

    rmse = np.sqrt(np.mean(np.square(errors)))

    bias = np.mean(errors)

    return {
        "samples": len(actual),
        "actual_mean": np.mean(actual),
        "predicted_mean": np.mean(predicted),
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
    }


# ---------------------------------------------------------------------
# Stockout severity grouping
# ---------------------------------------------------------------------

def get_stockout_group(
    stockout_hours: np.ndarray,
) -> np.ndarray:

    conditions = [
        stockout_hours == 0,
        (stockout_hours >= 1) & (stockout_hours <= 2),
        (stockout_hours >= 3) & (stockout_hours <= 5),
        (stockout_hours >= 6) & (stockout_hours <= 9),
        (stockout_hours >= 10) & (stockout_hours <= 15),
        stockout_hours == 16,
    ]

    labels = [
        "Fully in stock",
        "Low",
        "Moderate",
        "High",
        "Very high",
        "Fully out of stock",
    ]

    return np.select(
        conditions,
        labels,
        default="Invalid",
    )


# ---------------------------------------------------------------------
# Evaluate baselines
# ---------------------------------------------------------------------

def evaluate_split(
    df: pl.DataFrame,
    split_name: str,
    training_mean: float,
) -> list[dict]:

    actual = df[TARGET_COLUMN].to_numpy().astype(np.float64)

    observed = (
        df[OBSERVED_COLUMN]
        .to_numpy()
        .astype(np.float64)
    )

    stockout_hours = (
        df[STOCKOUT_HOURS_COLUMN]
        .to_numpy()
        .astype(np.int64)
    )

    if not (
        np.isfinite(actual).all()
        and np.isfinite(observed).all()
    ):
        raise ValueError(
            f"{split_name} contains non-finite target or prediction values."
        )

    if ((stockout_hours < 0) | (stockout_hours > 16)).any():
        raise ValueError(
            f"{split_name} contains invalid stockout-hour values."
        )

    severity_groups = get_stockout_group(stockout_hours)

    predictions = {
        "Observed-sales baseline": observed,
        "Training-mean baseline": np.full(
            len(actual),
            training_mean,
        ),
    }

    results = []

    for baseline_name, predicted in predictions.items():

        # Overall metrics.
        overall_metrics = calculate_metrics(
            actual,
            predicted,
        )

        results.append({
            "split": split_name,
            "baseline": baseline_name,
            "stockout_group": "Overall",
            **overall_metrics,
        })

        # Metrics for each stockout severity group.
        for group_name, _, _ in STOCKOUT_GROUPS:

            mask = severity_groups == group_name

            if not mask.any():
                continue

            group_metrics = calculate_metrics(
                actual[mask],
                predicted[mask],
            )

            results.append({
                "split": split_name,
                "baseline": baseline_name,
                "stockout_group": group_name,
                **group_metrics,
            })

    return results


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def print_results(
    metrics_df: pl.DataFrame,
) -> None:

    print("\n" + "=" * 95)
    print("DAILY DEMAND RECOVERY BASELINE RESULTS")
    print("=" * 95)

    for split_name in ["train", "validation"]:

        split_results = metrics_df.filter(
            pl.col("split") == split_name
        )

        print(f"\n{split_name.upper()} RESULTS")
        print("-" * 95)

        overall = split_results.filter(
            pl.col("stockout_group") == "Overall"
        )

        print(overall.select(
            "baseline",
            "samples",
            "actual_mean",
            "predicted_mean",
            "mae",
            "rmse",
            "bias",
        ))

        print("\nResults by stockout severity:")

        severity_results = split_results.filter(
            pl.col("stockout_group") != "Overall"
        )

        print(severity_results.select(
            "baseline",
            "stockout_group",
            "samples",
            "mae",
            "rmse",
            "bias",
        ))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    print("=" * 75)
    print("DAILY DEMAND RECOVERY BASELINE EVALUATION")
    print("=" * 75)

    # Load training data.
    train_df = load_dataset(
        TRAIN_FILE,
        "Training",
    )

    # Load validation data.
    validation_df = load_dataset(
        VALIDATION_FILE,
        "Validation",
    )

    # The training target mean is calculated only from training data.
    training_mean = float(
        train_df[TARGET_COLUMN].mean()
    )

    print("\nTraining target statistics")
    print("-" * 50)
    print(f"Training samples: {train_df.height:,}")
    print(f"Training mean daily demand: {training_mean:.6f}")

    # Evaluate both baselines on training and validation.
    results = []

    results.extend(
        evaluate_split(
            df=train_df,
            split_name="train",
            training_mean=training_mean,
        )
    )

    results.extend(
        evaluate_split(
            df=validation_df,
            split_name="validation",
            training_mean=training_mean,
        )
    )

    metrics_df = pl.DataFrame(results)

    print_results(metrics_df)

    # Save all metrics.
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_df.write_csv(OUTPUT_FILE)

    print("\n" + "=" * 75)
    print("BASELINE EVALUATION COMPLETE")
    print("=" * 75)
    print(f"Saved metrics: {OUTPUT_FILE}")
    print(f"Metric records: {metrics_df.height:,}")

    print("\nThe test dataset was not loaded or evaluated.")


if __name__ == "__main__":
    main()
