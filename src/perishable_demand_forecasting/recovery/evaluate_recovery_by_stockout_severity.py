
from pathlib import Path

import joblib
import numpy as np
import polars as pl

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

EXPERIMENT = "G_business_hourly"

VALIDATION_FILE = (
    Path("data/processed/freshretailnet/features/experiments")
    / EXPERIMENT
    / "validation.parquet"
)

MODEL_FILE = (
    Path("models/recovery/experiments")
    / f"{EXPERIMENT}.joblib"
)

OUTPUT_FILE = Path(
    "data/processed/freshretailnet/recovery/"
    "severity_evaluation.csv"
)

TARGET = "target_daily_demand"
OBSERVED = "simulated_observed_daily_sales"
SEVERITY = "simulated_stockout_hours"

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
    predicted: np.ndarray,
) -> dict:
    errors = predicted - actual

    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "bias": float(np.mean(errors)),
    }


def evaluate_group(
    group_name: str,
    actual: np.ndarray,
    model_predictions: np.ndarray,
    baseline_predictions: np.ndarray,
) -> dict:
    model_metrics = calculate_metrics(
        actual,
        model_predictions,
    )

    baseline_metrics = calculate_metrics(
        actual,
        baseline_predictions,
    )

    baseline_mae = baseline_metrics["mae"]

    if baseline_mae > 0:
        improvement_pct = (
            (baseline_mae - model_metrics["mae"])
            / baseline_mae
            * 100
        )
    else:
        improvement_pct = 0.0

    return {
        "severity": group_name,
        "observations": len(actual),
        "model_mae": model_metrics["mae"],
        "model_rmse": model_metrics["rmse"],
        "model_bias": model_metrics["bias"],
        "baseline_mae": baseline_metrics["mae"],
        "baseline_rmse": baseline_metrics["rmse"],
        "baseline_bias": baseline_metrics["bias"],
        "mae_improvement_pct": improvement_pct,
        "negative_prediction_pct": float(
            np.mean(model_predictions < 0) * 100
        ),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print("=" * 75)
    print("DAILY DEMAND RECOVERY: STOCKOUT SEVERITY EVALUATION")
    print("=" * 75)

    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            f"Validation dataset not found: {VALIDATION_FILE}"
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_FILE}"
        )

    print("\nLoading validation dataset...")
    df = pl.read_parquet(VALIDATION_FILE)

    required_columns = {
        TARGET,
        OBSERVED,
        SEVERITY,
        "dt",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df.height == 0:
        raise ValueError("Validation dataset is empty.")

    print(f"Validation rows: {df.height:,}")

    print("\nLoading trained model...")
    model = joblib.load(MODEL_FILE)

    feature_columns = [
        column
        for column in df.columns
        if column not in {TARGET, "dt"}
    ]

    X = df.select(feature_columns).to_pandas()

    actual = df[TARGET].to_numpy()
    baseline_predictions = df[OBSERVED].to_numpy()
    stockout_hours = df[SEVERITY].to_numpy()

    print("\nGenerating model predictions...")
    model_predictions = model.predict(X)

    if not np.isfinite(model_predictions).all():
        raise ValueError(
            "Model generated non-finite predictions."
        )

    results = []

    # Overall evaluation
    results.append(
        evaluate_group(
            "Overall",
            actual,
            model_predictions,
            baseline_predictions,
        )
    )

    # Severity-level evaluation
    for group_name, minimum, maximum in SEVERITY_GROUPS:
        mask = (
            (stockout_hours >= minimum)
            & (stockout_hours <= maximum)
        )

        if not mask.any():
            print(
                f"\nSkipping {group_name}: no observations."
            )
            continue

        results.append(
            evaluate_group(
                group_name,
                actual[mask],
                model_predictions[mask],
                baseline_predictions[mask],
            )
        )

    results_df = pl.DataFrame(results)

    print("\n" + "=" * 75)
    print("RECOVERY PERFORMANCE BY STOCKOUT SEVERITY")
    print("=" * 75)

    print(results_df)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.write_csv(OUTPUT_FILE)

    print(f"\nResults saved: {OUTPUT_FILE}")

    print("\n" + "=" * 75)
    print("SEVERITY EVALUATION COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()
