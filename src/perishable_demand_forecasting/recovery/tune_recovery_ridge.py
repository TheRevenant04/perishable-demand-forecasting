
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

EXPERIMENT = "G_business_hourly"

ALPHA_VALUES = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]

FEATURE_DIR = Path("data/processed/freshretailnet/features/experiments")
RECOVERY_DIR = Path("data/processed/freshretailnet/recovery")
MODEL_DIR = Path("models/recovery/regularization")

TARGET = "target_daily_demand"
OBSERVED_BASELINE = "simulated_observed_daily_sales"

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

def load_dataset(split: str) -> pl.DataFrame:
    path = FEATURE_DIR / EXPERIMENT / f"{split}.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {split} dataset: {path}\n"
            "Run perishable_demand_forecasting.recovery.select_recovery_features first."
        )

    return pl.read_parquet(path)


def validate_datasets(
    train_df: pl.DataFrame,
    validation_df: pl.DataFrame,
) -> None:
    required_columns = {
        TARGET,
        OBSERVED_BASELINE,
        "dt",
    }

    for split, df in [
        ("train", train_df),
        ("validation", validation_df),
    ]:
        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"{split} dataset is missing columns: "
                f"{sorted(missing)}"
            )

        if df.height == 0:
            raise ValueError(
                f"{split} dataset is empty."
            )

        for column in [TARGET, OBSERVED_BASELINE]:
            values = df[column].to_numpy()

            if not np.isfinite(values).all():
                raise ValueError(
                    f"{split} contains null or non-finite "
                    f"values in {column}."
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
            "Training and validation feature columns differ."
        )


# ---------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------

def prepare_features(
    df: pl.DataFrame,
) -> tuple[pl.DataFrame, list[str], list[str]]:
    excluded_columns = {
        TARGET,
        "dt",
    }

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
    alpha: float,
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
            ("model", Ridge(alpha=alpha)),
        ]
    )


# ---------------------------------------------------------------------
# Baseline evaluation
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
                "alpha": None,
                "split": "validation",
                **metrics,
            }
        )

    return results


# ---------------------------------------------------------------------
# Ridge regularization experiments
# ---------------------------------------------------------------------

def train_alpha_experiment(
    alpha: float,
    X_train,
    X_validation,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    categorical_features: list[str],
    numerical_features: list[str],
) -> dict:
    print("\n" + "=" * 75)
    print(f"RIDGE REGULARIZATION: ALPHA = {alpha}")
    print("=" * 75)

    model = create_model(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        alpha=alpha,
    )

    model.fit(
        X_train,
        y_train,
    )

    train_predictions = model.predict(X_train)
    validation_predictions = model.predict(X_validation)

    train_metrics = calculate_metrics(
        y_train,
        train_predictions,
    )

    validation_metrics = calculate_metrics(
        y_validation,
        validation_predictions,
    )

    print_metrics("Train", train_metrics)
    print_metrics("Validation", validation_metrics)

    encoded_features = model.named_steps[
        "preprocessor"
    ].transform(X_train).shape[1]

    print(f"Encoded features: {encoded_features:,}")

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        MODEL_DIR / f"G_business_hourly_alpha_{alpha:g}.joblib"
    )

    joblib.dump(model, model_path)

    print(f"Saved model: {model_path}")

    return {
        "experiment": EXPERIMENT,
        "alpha": alpha,
        "split": "validation",
        **validation_metrics,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print("=" * 75)
    print("DAILY DEMAND RECOVERY: RIDGE REGULARIZATION EXPERIMENTS")
    print("=" * 75)

    print("\nLoading datasets...")

    train_df = load_dataset("train")
    validation_df = load_dataset("validation")

    validate_datasets(
        train_df,
        validation_df,
    )

    X_train, categorical_features, numerical_features = (
        prepare_features(train_df)
    )

    X_validation, _, _ = prepare_features(validation_df)

    X_train = X_train.to_pandas()
    X_validation = X_validation.to_pandas()

    y_train = train_df[TARGET].to_numpy()
    y_validation = validation_df[TARGET].to_numpy()

    print(f"\nTraining rows: {len(y_train):,}")
    print(f"Validation rows: {len(y_validation):,}")
    print(f"Original features: {X_train.shape[1]}")
    print(f"Categorical features: {len(categorical_features)}")
    print(f"Numerical features: {len(numerical_features)}")
    print(f"Alpha values: {ALPHA_VALUES}")

    print("\n" + "=" * 75)
    print("VALIDATION BASELINES")
    print("=" * 75)

    results = evaluate_baselines(
        train_df,
        validation_df,
    )

    for alpha in ALPHA_VALUES:
        result = train_alpha_experiment(
            alpha=alpha,
            X_train=X_train,
            X_validation=X_validation,
            y_train=y_train,
            y_validation=y_validation,
            categorical_features=categorical_features,
            numerical_features=numerical_features,
        )

        results.append(result)

    metrics_df = pl.DataFrame(results).sort(
        "mae",
        descending=False,
    )

    print("\n" + "=" * 75)
    print("RIDGE REGULARIZATION RESULTS")
    print("=" * 75)

    print(metrics_df)

    metrics_path = (
        RECOVERY_DIR / "ridge_alpha_metrics.csv"
    )

    RECOVERY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_df.write_csv(metrics_path)

    print(f"\nMetrics saved: {metrics_path}")

    print("\n" + "=" * 75)
    print("RIDGE REGULARIZATION EXPERIMENTS COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()
