
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

EXPERIMENTS = [
    "A_core",
    "B_product",
    "C_business",
    "D_weather",
    "E_hourly",
    "F_management",
    "G_business_hourly",
    "H_all_features",
]

FEATURE_DIR = Path("data/processed/freshretailnet/features/experiments")
RECOVERY_DIR = Path("data/processed/freshretailnet/recovery")
MODEL_DIR = Path("models/recovery/experiments")

TARGET = "target_daily_demand"
OBSERVED_BASELINE = "simulated_observed_daily_sales"

RIDGE_ALPHA = 1.0

CATEGORICAL_DTYPES = (pl.String, pl.Categorical)


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    errors = y_pred - y_true

    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "bias": float(np.mean(errors)),
        "r2": float(
            1 - np.sum(errors**2)
            / np.sum((y_true - np.mean(y_true)) ** 2)
        ),
        "negative_prediction_pct": float(
            np.mean(y_pred < 0) * 100
        ),
    }


def print_metrics(
    split: str,
    metrics: dict,
) -> None:
    print(
        f"{split:<12} "
        f"MAE: {metrics['mae']:.6f} | "
        f"RMSE: {metrics['rmse']:.6f} | "
        f"Bias: {metrics['bias']:.6f} | "
        f"R2: {metrics['r2']:.6f} | "
        f"Negative predictions: "
        f"{metrics['negative_prediction_pct']:.2f}%"
    )


# ---------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------

def load_experiment(
    experiment: str,
    split: str,
) -> pl.DataFrame:
    path = FEATURE_DIR / experiment / f"{split}.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {split} dataset for {experiment}: {path}\n"
            "Run perishable_demand_forecasting.recovery.select_recovery_features first."
        )

    return pl.read_parquet(path)


def validate_dataset(
    experiment: str,
    train_df: pl.DataFrame,
    validation_df: pl.DataFrame,
) -> None:
    required_columns = {TARGET, OBSERVED_BASELINE, "dt"}

    for split, df in [
        ("train", train_df),
        ("validation", validation_df),
    ]:
        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"{experiment} {split} is missing required "
                f"columns: {sorted(missing)}"
            )

        if df.height == 0:
            raise ValueError(
                f"{experiment} {split} dataset is empty."
            )

        if df[TARGET].null_count() > 0:
            raise ValueError(
                f"{experiment} {split} contains null targets."
            )

        if df[OBSERVED_BASELINE].null_count() > 0:
            raise ValueError(
                f"{experiment} {split} contains null baseline values."
            )

        target_values = df[TARGET].to_numpy()

        if not np.isfinite(target_values).all():
            raise ValueError(
                f"{experiment} {split} contains non-finite targets."
            )

        baseline_values = df[OBSERVED_BASELINE].to_numpy()

        if not np.isfinite(baseline_values).all():
            raise ValueError(
                f"{experiment} {split} contains non-finite baseline values."
            )

    train_features = set(train_df.columns) - {
        TARGET,
        "dt",
    }

    validation_features = set(validation_df.columns) - {
        TARGET,
        "dt",
    }

    if train_features != validation_features:
        raise ValueError(
            f"{experiment} train and validation features differ."
        )


# ---------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------

def prepare_features(
    df: pl.DataFrame,
) -> tuple[pl.DataFrame, list[str], list[str]]:
    excluded_columns = {TARGET, "dt"}

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    categorical_features = [
        column
        for column in feature_columns
        if df.schema[column] in CATEGORICAL_DTYPES
    ]

    numerical_features = [
        column
        for column in feature_columns
        if column not in categorical_features
    ]

    return (
        df.select(feature_columns),
        categorical_features,
        numerical_features,
    )


def create_one_hot_encoder():
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=True,
        )


def create_model(
    categorical_features: list[str],
    numerical_features: list[str],
) -> Pipeline:
    transformers = []

    if categorical_features:
        transformers.append(
            (
                "categorical",
                create_one_hot_encoder(),
                categorical_features,
            )
        )

    if numerical_features:
        transformers.append(
            (
                "numerical",
                StandardScaler(),
                numerical_features,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=RIDGE_ALPHA)),
        ]
    )


# ---------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------

def evaluate_baselines(
    train_df: pl.DataFrame,
    validation_df: pl.DataFrame,
) -> list[dict]:
    y_validation = validation_df[TARGET].to_numpy()

    observed_predictions = (
        validation_df[OBSERVED_BASELINE].to_numpy()
    )

    training_mean = float(train_df[TARGET].mean())

    mean_predictions = np.full(
        validation_df.height,
        training_mean,
    )

    results = []

    for name, predictions in [
        ("observed_sales_baseline", observed_predictions),
        ("training_mean_baseline", mean_predictions),
    ]:
        metrics = calculate_metrics(
            y_validation,
            predictions,
        )

        print_metrics("Validation", metrics)

        results.append(
            {
                "experiment": name,
                "split": "validation",
                **metrics,
            }
        )

    return results


# ---------------------------------------------------------------------
# Experiment training
# ---------------------------------------------------------------------

def train_experiment(
    experiment: str,
) -> dict:
    print("\n" + "=" * 75)
    print(f"TRAINING EXPERIMENT: {experiment}")
    print("=" * 75)

    train_df = load_experiment(experiment, "train")
    validation_df = load_experiment(experiment, "validation")

    validate_dataset(
        experiment,
        train_df,
        validation_df,
    )

    X_train, categorical_features, numerical_features = (
        prepare_features(train_df)
    )

    X_validation, _, _ = prepare_features(validation_df)

    y_train = train_df[TARGET].to_numpy()
    y_validation = validation_df[TARGET].to_numpy()

    print(f"Training rows: {train_df.height:,}")
    print(f"Validation rows: {validation_df.height:,}")
    print(f"Original features: {X_train.width}")
    print(f"Categorical features: {len(categorical_features)}")
    print(f"Numerical features: {len(numerical_features)}")
    print(f"Ridge alpha: {RIDGE_ALPHA}")

    model = create_model(
        categorical_features,
        numerical_features,
    )

    X_train_pandas = X_train.to_pandas()
    X_validation_pandas = X_validation.to_pandas()

    model.fit(
        X_train_pandas,
        y_train,
    )

    train_predictions = model.predict(
        X_train_pandas
    )

    validation_predictions = model.predict(
        X_validation_pandas
    )

    train_metrics = calculate_metrics(
        y_train,
        train_predictions,
    )

    validation_metrics = calculate_metrics(
        y_validation,
        validation_predictions,
    )

    encoded_features = model.named_steps[
        "preprocessor"
    ].transform(
        X_train_pandas
    ).shape[1]

    print(f"Encoded features: {encoded_features:,}")
    print_metrics("Train", train_metrics)
    print_metrics("Validation", validation_metrics)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = MODEL_DIR / f"{experiment}.joblib"

    joblib.dump(model, model_path)

    print(f"Saved model: {model_path}")

    return {
        "experiment": experiment,
        "split": "validation",
        **validation_metrics,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print("=" * 75)
    print("DAILY DEMAND RECOVERY: CONTROLLED MODEL EXPERIMENTS")
    print("=" * 75)

    print("\nLoading baseline datasets...")

    baseline_train = load_experiment("A_core", "train")
    baseline_validation = load_experiment(
        "A_core",
        "validation",
    )

    validate_dataset(
        "A_core",
        baseline_train,
        baseline_validation,
    )

    print("\n" + "=" * 75)
    print("VALIDATION BASELINES")
    print("=" * 75)

    results = evaluate_baselines(
        baseline_train,
        baseline_validation,
    )

    for experiment in EXPERIMENTS:
        result = train_experiment(experiment)
        results.append(result)

    metrics_df = pl.DataFrame(results).sort(
        "mae",
        descending=False,
    )

    print("\n" + "=" * 75)
    print("CONTROLLED EXPERIMENT RESULTS")
    print("=" * 75)

    print(metrics_df)

    metrics_path = (
        RECOVERY_DIR / "experiment_metrics.csv"
    )

    RECOVERY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_df.write_csv(metrics_path)

    print(f"\nMetrics saved: {metrics_path}")

    print("\n" + "=" * 75)
    print("MODEL EXPERIMENTS COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()
