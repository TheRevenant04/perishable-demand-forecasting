
from pathlib import Path

import joblib
import numpy as np
import polars as pl

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

EXPERIMENT = "G_business_hourly"

FEATURE_DIR = Path(
    "data/processed/freshretailnet/features/experiments"
)

MODEL_DIR = Path("models/recovery/regularization")

OUTPUT_DIR = Path("data/processed/freshretailnet/recovery")

TARGET = "target_daily_demand"
OBSERVED_BASELINE = "simulated_observed_daily_sales"
STOCKOUT_HOURS = "simulated_stockout_hours"

MODELS = {
    "ridge_alpha_1": MODEL_DIR
    / "G_business_hourly_alpha_1.joblib",
    "ridge_alpha_10": MODEL_DIR
    / "G_business_hourly_alpha_10.joblib",
}

SEVERITY_GROUPS = [
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
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    errors = y_pred - y_true

    return {
        "rows": len(y_true),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "bias": float(np.mean(errors)),
        "negative_predictions": int(np.sum(y_pred < 0)),
        "negative_prediction_pct": float(
            np.mean(y_pred < 0) * 100
        ),
        "minimum_prediction": float(np.min(y_pred)),
    }


def print_metrics(
    name: str,
    metrics: dict,
) -> None:
    print(
        f"{name:<22}"
        f"Rows: {metrics['rows']:>7,} | "
        f"MAE: {metrics['mae']:.6f} | "
        f"RMSE: {metrics['rmse']:.6f} | "
        f"Bias: {metrics['bias']:.6f} | "
        f"Negative: {metrics['negative_prediction_pct']:.2f}%"
    )


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_validation_data() -> pl.DataFrame:
    path = FEATURE_DIR / EXPERIMENT / "validation.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"Validation dataset not found: {path}"
        )

    df = pl.read_parquet(path)

    required_columns = {
        TARGET,
        OBSERVED_BASELINE,
        STOCKOUT_HOURS,
        "dt",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Validation dataset is missing columns: "
            f"{sorted(missing)}"
        )

    return df


def load_models() -> dict:
    models = {}

    for name, path in MODELS.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {path}"
            )

        models[name] = joblib.load(path)

    return models


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------

def generate_predictions(
    df: pl.DataFrame,
    models: dict,
) -> dict:
    excluded_columns = {
        TARGET,
        "dt",
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    X = df.select(feature_columns).to_pandas()

    predictions = {
        "observed_sales_baseline": (
            df[OBSERVED_BASELINE].to_numpy()
        )
    }

    for name, model in models.items():
        predictions[name] = model.predict(X)

    return predictions


# ---------------------------------------------------------------------
# Severity evaluation
# ---------------------------------------------------------------------

def evaluate_severity(
    df: pl.DataFrame,
    predictions: dict,
) -> pl.DataFrame:
    y_true = df[TARGET].to_numpy()
    stockout_hours = df[STOCKOUT_HOURS].to_numpy()

    results = []

    groups = [
        ("1-2 hours", 1, 2),
        ("3-5 hours", 3, 5),
        ("6-9 hours", 6, 9),
        ("10-15 hours", 10, 15),
        ("16 hours", 16, 16),
    ]

    for group_name, lower, upper in groups:
        mask = (
            (stockout_hours >= lower)
            & (stockout_hours <= upper)
        )

        if not np.any(mask):
            print(
                f"\nSkipping {group_name}: no observations."
            )
            continue

        group_results = {}

        print("\n" + "-" * 75)
        print(
            f"STOCKOUT SEVERITY: {group_name} "
            f"({np.sum(mask):,} rows)"
        )
        print("-" * 75)

        for model_name, model_predictions in predictions.items():
            metrics = calculate_metrics(
                y_true[mask],
                model_predictions[mask],
            )

            print_metrics(
                model_name,
                metrics,
            )

            group_results[model_name] = metrics

        baseline_mae = group_results[
            "observed_sales_baseline"
        ]["mae"]

        for model_name in [
            "ridge_alpha_1",
            "ridge_alpha_10",
        ]:
            metrics = group_results[model_name]

            improvement = (
                (baseline_mae - metrics["mae"])
                / baseline_mae
                * 100
                if baseline_mae > 0
                else 0.0
            )

            results.append(
                {
                    "severity": group_name,
                    "model": model_name,
                    **metrics,
                    "baseline_mae": baseline_mae,
                    "improvement_vs_baseline_pct": improvement,
                }
            )

    return pl.DataFrame(results)


# ---------------------------------------------------------------------
# Negative prediction analysis
# ---------------------------------------------------------------------

def analyse_negative_predictions(
    df: pl.DataFrame,
    predictions: dict,
) -> pl.DataFrame:
    stockout_hours = df[STOCKOUT_HOURS].to_numpy()
    y_true = df[TARGET].to_numpy()

    results = []

    print("\n" + "=" * 75)
    print("NEGATIVE PREDICTION ANALYSIS")
    print("=" * 75)

    for model_name in [
        "ridge_alpha_1",
        "ridge_alpha_10",
    ]:
        model_predictions = predictions[model_name]

        negative_mask = model_predictions < 0

        negative_count = int(np.sum(negative_mask))

        print(f"\n{model_name}")
        print(f"Negative predictions: {negative_count:,}")
        print(
            f"Negative prediction rate: "
            f"{np.mean(negative_mask) * 100:.2f}%"
        )

        if negative_count:
            print(
                f"Most negative prediction: "
                f"{np.min(model_predictions):.6f}"
            )

            print(
                f"Average actual target for negative predictions: "
                f"{np.mean(y_true[negative_mask]):.6f}"
            )

        for group_name, lower, upper in SEVERITY_GROUPS:
            group_mask = (
                (stockout_hours >= lower)
                & (stockout_hours <= upper)
            )

            group_negative_mask = (
                group_mask & negative_mask
            )

            group_rows = int(np.sum(group_mask))
            group_negative = int(
                np.sum(group_negative_mask)
            )

            results.append(
                {
                    "model": model_name,
                    "severity": group_name,
                    "rows": group_rows,
                    "negative_predictions": group_negative,
                    "negative_prediction_pct": (
                        group_negative / group_rows * 100
                        if group_rows
                        else 0.0
                    ),
                }
            )

    return pl.DataFrame(results)


# ---------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------

def save_results(
    severity_df: pl.DataFrame,
    negative_df: pl.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    severity_path = (
        OUTPUT_DIR / "ridge_alpha_severity_evaluation.csv"
    )

    negative_path = (
        OUTPUT_DIR / "ridge_alpha_negative_predictions.csv"
    )

    severity_df.write_csv(severity_path)
    negative_df.write_csv(negative_path)

    print(f"\nSeverity results saved: {severity_path}")
    print(f"Negative prediction results saved: {negative_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print("=" * 75)
    print("DAILY DEMAND RECOVERY: RIDGE SEVERITY EVALUATION")
    print("=" * 75)

    print("\nLoading validation data...")
    df = load_validation_data()

    print(f"Validation rows: {df.height:,}")

    print("\nLoading Ridge models...")
    models = load_models()

    print("\nGenerating predictions...")
    predictions = generate_predictions(
        df,
        models,
    )

    print("\n" + "=" * 75)
    print("OVERALL VALIDATION RESULTS")
    print("=" * 75)

    y_true = df[TARGET].to_numpy()

    for model_name, model_predictions in predictions.items():
        metrics = calculate_metrics(
            y_true,
            model_predictions,
        )

        print_metrics(
            model_name,
            metrics,
        )

    severity_df = evaluate_severity(
        df,
        predictions,
    )

    negative_df = analyse_negative_predictions(
        df,
        predictions,
    )

    print("\n" + "=" * 75)
    print("NEGATIVE PREDICTIONS BY SEVERITY")
    print("=" * 75)

    print(negative_df)

    save_results(
        severity_df,
        negative_df,
    )

    print("\n" + "=" * 75)
    print("RIDGE SEVERITY EVALUATION COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()
