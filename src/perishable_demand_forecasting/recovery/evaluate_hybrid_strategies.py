
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

MODEL_PATHS = {
    "ridge_alpha_1": (
        MODEL_DIR / "G_business_hourly_alpha_1.joblib"
    ),
    "ridge_alpha_10": (
        MODEL_DIR / "G_business_hourly_alpha_10.joblib"
    ),
}


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
    }


def print_metrics(
    name: str,
    metrics: dict,
) -> None:
    print(
        f"{name:<25}"
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
            f"Missing required columns: {sorted(missing)}"
        )

    if df.height == 0:
        raise ValueError("Validation dataset is empty.")

    return df


def load_models() -> dict:
    models = {}

    for name, path in MODEL_PATHS.items():
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
        ),
    }

    for name, model in models.items():
        predictions[name] = model.predict(X)

    return predictions


# ---------------------------------------------------------------------
# Hybrid strategy
# ---------------------------------------------------------------------

def create_hybrid_predictions(
    df: pl.DataFrame,
    predictions: dict,
) -> np.ndarray:
    stockout_hours = df[STOCKOUT_HOURS].to_numpy()

    observed_predictions = predictions[
        "observed_sales_baseline"
    ]

    alpha_1_predictions = predictions[
        "ridge_alpha_1"
    ]

    alpha_10_predictions = predictions[
        "ridge_alpha_10"
    ]

    hybrid_predictions = np.empty(
        df.height,
        dtype=float,
    )

    # Light stockouts: retain observed daily sales.
    light_stockout_mask = (
        (stockout_hours >= 1)
        & (stockout_hours <= 5)
    )

    # Moderate stockouts: use Ridge alpha 10.
    moderate_stockout_mask = (
        (stockout_hours >= 6)
        & (stockout_hours <= 9)
    )

    # Severe stockouts: use Ridge alpha 1.
    severe_stockout_mask = (
        (stockout_hours >= 10)
        & (stockout_hours <= 16)
    )

    hybrid_predictions[light_stockout_mask] = (
        observed_predictions[light_stockout_mask]
    )

    hybrid_predictions[moderate_stockout_mask] = (
        alpha_10_predictions[moderate_stockout_mask]
    )

    hybrid_predictions[severe_stockout_mask] = (
        alpha_1_predictions[severe_stockout_mask]
    )

    # The current recovery dataset should contain only 1–16
    # simulated stockout hours. Fail rather than silently
    # assigning predictions to unexpected records.
    assigned_mask = (
        light_stockout_mask
        | moderate_stockout_mask
        | severe_stockout_mask
    )

    if not np.all(assigned_mask):
        unexpected_hours = np.unique(
            stockout_hours[~assigned_mask]
        )

        raise ValueError(
            "Unexpected simulated stockout-hour values: "
            f"{unexpected_hours.tolist()}"
        )

    return hybrid_predictions


# ---------------------------------------------------------------------
# Overall comparison
# ---------------------------------------------------------------------

def evaluate_overall(
    df: pl.DataFrame,
    predictions: dict,
) -> pl.DataFrame:
    y_true = df[TARGET].to_numpy()

    results = []

    print("\n" + "=" * 75)
    print("OVERALL VALIDATION COMPARISON")
    print("=" * 75)

    for name, y_pred in predictions.items():
        metrics = calculate_metrics(
            y_true,
            y_pred,
        )

        print_metrics(name, metrics)

        results.append(
            {
                "strategy": name,
                **metrics,
            }
        )

    return pl.DataFrame(results).sort("mae")


# ---------------------------------------------------------------------
# Severity comparison
# ---------------------------------------------------------------------

def evaluate_by_severity(
    df: pl.DataFrame,
    predictions: dict,
) -> pl.DataFrame:
    y_true = df[TARGET].to_numpy()
    stockout_hours = df[STOCKOUT_HOURS].to_numpy()

    groups = [
        ("1-2 hours", 1, 2),
        ("3-5 hours", 3, 5),
        ("6-9 hours", 6, 9),
        ("10-15 hours", 10, 15),
        ("16 hours", 16, 16),
    ]

    results = []

    print("\n" + "=" * 75)
    print("VALIDATION RESULTS BY STOCKOUT SEVERITY")
    print("=" * 75)

    for group_name, lower, upper in groups:
        mask = (
            (stockout_hours >= lower)
            & (stockout_hours <= upper)
        )

        if not np.any(mask):
            continue

        print(
            f"\n{group_name} "
            f"({int(np.sum(mask)):,} rows)"
        )

        for strategy_name, y_pred in predictions.items():
            metrics = calculate_metrics(
                y_true[mask],
                y_pred[mask],
            )

            print_metrics(
                strategy_name,
                metrics,
            )

            results.append(
                {
                    "severity": group_name,
                    "strategy": strategy_name,
                    **metrics,
                }
            )

    return pl.DataFrame(results)


# ---------------------------------------------------------------------
# Hybrid routing distribution
# ---------------------------------------------------------------------

def evaluate_hybrid_distribution(
    df: pl.DataFrame,
) -> pl.DataFrame:
    stockout_hours = df[STOCKOUT_HOURS].to_numpy()

    groups = [
        ("observed_sales", 1, 5),
        ("ridge_alpha_10", 6, 9),
        ("ridge_alpha_1", 10, 16),
    ]

    results = []

    print("\n" + "=" * 75)
    print("HYBRID STRATEGY ROUTING DISTRIBUTION")
    print("=" * 75)

    for strategy, lower, upper in groups:
        count = int(
            np.sum(
                (stockout_hours >= lower)
                & (stockout_hours <= upper)
            )
        )

        percentage = count / len(stockout_hours) * 100

        print(
            f"{strategy:<22}"
            f"Rows: {count:>7,} | "
            f"Percentage: {percentage:.2f}%"
        )

        results.append(
            {
                "strategy": strategy,
                "rows": count,
                "percentage": percentage,
            }
        )

    return pl.DataFrame(results)


# ---------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------

def save_results(
    overall_df: pl.DataFrame,
    severity_df: pl.DataFrame,
    distribution_df: pl.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "hybrid_recovery_overall.csv": overall_df,
        "hybrid_recovery_severity.csv": severity_df,
        "hybrid_recovery_distribution.csv": distribution_df,
    }

    for filename, result_df in outputs.items():
        path = OUTPUT_DIR / filename
        result_df.write_csv(path)

        print(f"Saved: {path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print("=" * 75)
    print("DAILY DEMAND RECOVERY: HYBRID STRATEGY EVALUATION")
    print("=" * 75)

    print("\nLoading validation data...")
    df = load_validation_data()

    print(f"Validation rows: {df.height:,}")

    print("\nLoading trained models...")
    models = load_models()

    print("\nGenerating predictions...")
    predictions = generate_predictions(
        df,
        models,
    )

    print("\nCreating hybrid predictions...")
    predictions["hybrid_recovery"] = (
        create_hybrid_predictions(
            df,
            predictions,
        )
    )

    overall_df = evaluate_overall(
        df,
        predictions,
    )

    severity_df = evaluate_by_severity(
        df,
        predictions,
    )

    distribution_df = evaluate_hybrid_distribution(df)

    save_results(
        overall_df,
        severity_df,
        distribution_df,
    )

    print("\n" + "=" * 75)
    print("HYBRID RECOVERY EVALUATION COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()
