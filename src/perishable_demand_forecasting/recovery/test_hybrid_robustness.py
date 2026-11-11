
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
OBSERVED = "simulated_observed_daily_sales"
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
# Hybrid strategies
# ---------------------------------------------------------------------

STRATEGIES = {
    "current_hybrid": {
        "1-5": "observed",
        "6-9": "ridge_alpha_10",
        "10-16": "ridge_alpha_1",
    },
    "alternative_a": {
        "1-5": "observed",
        "6-9": "ridge_alpha_1",
        "10-16": "ridge_alpha_1",
    },
    "alternative_b": {
        "1-5": "observed",
        "6-9": "ridge_alpha_10",
        "10-16": "ridge_alpha_10",
    },
    "alternative_c": {
        "1-5": "observed",
        "6-9": "ridge_alpha_1",
        "10-16": "ridge_alpha_10",
    },
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
        f"{name:<24}"
        f"MAE: {metrics['mae']:.6f} | "
        f"RMSE: {metrics['rmse']:.6f} | "
        f"Bias: {metrics['bias']:.6f} | "
        f"Negative: {metrics['negative_prediction_pct']:.2f}%"
    )


# ---------------------------------------------------------------------
# Load data and models
# ---------------------------------------------------------------------

def load_validation_data() -> pl.DataFrame:
    path = FEATURE_DIR / EXPERIMENT / "validation.parquet"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pl.read_parquet(path)

    required = {
        TARGET,
        OBSERVED,
        STOCKOUT_HOURS,
        "dt",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing validation columns: {sorted(missing)}"
        )

    if df.height == 0:
        raise ValueError("Validation dataset is empty.")

    hours = df[STOCKOUT_HOURS].to_numpy()

    if not np.isfinite(hours).all():
        raise ValueError("Invalid stockout-hour values.")

    if not np.all((hours >= 1) & (hours <= 16)):
        raise ValueError(
            "Expected simulated stockout hours between 1 and 16."
        )

    return df


def load_models() -> dict:
    models = {}

    for name, path in MODEL_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(path)

        models[name] = joblib.load(path)

    return models


# ---------------------------------------------------------------------
# Generate predictions
# ---------------------------------------------------------------------

def generate_predictions(
    df: pl.DataFrame,
    models: dict,
) -> dict:
    excluded = {
        TARGET,
        "dt",
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded
    ]

    X = df.select(feature_columns).to_pandas()

    predictions = {
        "observed": df[OBSERVED].to_numpy(),
    }

    for name, model in models.items():
        predictions[name] = model.predict(X)

    return predictions


# ---------------------------------------------------------------------
# Create hybrid predictions
# ---------------------------------------------------------------------

def create_hybrid_predictions(
    stockout_hours: np.ndarray,
    predictions: dict,
    strategy: dict,
) -> np.ndarray:
    result = np.empty(
        len(stockout_hours),
        dtype=float,
    )

    groups = [
        (1, 5, strategy["1-5"]),
        (6, 9, strategy["6-9"]),
        (10, 16, strategy["10-16"]),
    ]

    assigned = np.zeros(
        len(stockout_hours),
        dtype=bool,
    )

    for lower, upper, source in groups:
        mask = (
            (stockout_hours >= lower)
            & (stockout_hours <= upper)
        )

        if np.any(assigned & mask):
            raise ValueError("Overlapping routing groups.")

        result[mask] = predictions[source][mask]
        assigned[mask] = True

    if not assigned.all():
        raise ValueError(
            "Some validation observations were not assigned."
        )

    return result


# ---------------------------------------------------------------------
# Overall evaluation
# ---------------------------------------------------------------------

def evaluate_overall(
    y_true: np.ndarray,
    predictions: dict,
) -> pl.DataFrame:
    results = []

    print("\n" + "=" * 85)
    print("OVERALL VALIDATION COMPARISON")
    print("=" * 85)

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
# Severity evaluation
# ---------------------------------------------------------------------

def evaluate_severity(
    y_true: np.ndarray,
    stockout_hours: np.ndarray,
    predictions: dict,
) -> pl.DataFrame:
    groups = [
        ("1-2 hours", 1, 2),
        ("3-5 hours", 3, 5),
        ("6-9 hours", 6, 9),
        ("10-15 hours", 10, 15),
        ("16 hours", 16, 16),
    ]

    results = []

    print("\n" + "=" * 85)
    print("VALIDATION RESULTS BY STOCKOUT SEVERITY")
    print("=" * 85)

    for group_name, lower, upper in groups:
        mask = (
            (stockout_hours >= lower)
            & (stockout_hours <= upper)
        )

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
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print("=" * 85)
    print("DAILY DEMAND RECOVERY: HYBRID ROBUSTNESS EXPERIMENT")
    print("=" * 85)

    print("\nLoading validation data...")
    df = load_validation_data()

    print(f"Validation rows: {df.height:,}")

    print("\nLoading trained models...")
    models = load_models()

    print("\nGenerating model predictions...")
    base_predictions = generate_predictions(
        df,
        models,
    )

    all_predictions = {
        "observed_sales_baseline": base_predictions["observed"],
        "ridge_alpha_1": base_predictions["ridge_alpha_1"],
        "ridge_alpha_10": base_predictions["ridge_alpha_10"],
    }

    stockout_hours = df[STOCKOUT_HOURS].to_numpy()

    print("\nEvaluating hybrid strategies...")

    for strategy_name, routing in STRATEGIES.items():
        all_predictions[strategy_name] = (
            create_hybrid_predictions(
                stockout_hours,
                base_predictions,
                routing,
            )
        )

    y_true = df[TARGET].to_numpy()

    overall_df = evaluate_overall(
        y_true,
        all_predictions,
    )

    severity_df = evaluate_severity(
        y_true,
        stockout_hours,
        all_predictions,
    )

    # -----------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall_path = (
        OUTPUT_DIR / "hybrid_robustness_overall.csv"
    )

    severity_path = (
        OUTPUT_DIR / "hybrid_robustness_severity.csv"
    )

    overall_df.write_csv(overall_path)
    severity_df.write_csv(severity_path)

    print("\n" + "=" * 85)
    print("RESULTS SAVED")
    print("=" * 85)

    print(f"Overall: {overall_path}")
    print(f"Severity: {severity_path}")

    print("\n" + "=" * 85)
    print("HYBRID ROBUSTNESS EXPERIMENT COMPLETE")
    print("=" * 85)


if __name__ == "__main__":
    main()
