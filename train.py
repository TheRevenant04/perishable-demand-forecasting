"""Train the final demand forecasting model.

The final model is a Random Forest regressor selected during the
tree-based model comparison. It is fitted on the chronological
train + validation splits and saved as a pickle file for serving.

Usage:
    uv run python train.py
"""

from datetime import datetime, timezone
from pathlib import Path

import pickle
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor

from perishable_demand_forecasting.config import FORECASTING_DIR, FORECASTING_MODELS_DIR

TRAIN_PATH = FORECASTING_DIR / "train.parquet"
VALIDATION_PATH = FORECASTING_DIR / "validation.parquet"
MODEL_PATH = FORECASTING_MODELS_DIR / "final_demand_forecast.pkl"

TARGET_COLUMN = "target_daily_demand"
HORIZON_COLUMN = "forecast_horizon"

HISTORICAL_FEATURES = (
    [f"demand_lag_{i}" for i in range(1, 15)]
    + [
        "demand_rolling_mean_7",
        "demand_rolling_mean_14",
        "stockout_hours_lag_1",
        "stockout_hours_lag_7",
    ]
)

FUTURE_FEATURES = [
    HORIZON_COLUMN,
    "target_weekday",
    "target_month",
    "target_holiday_flag",
]

FEATURES = HISTORICAL_FEATURES + FUTURE_FEATURES

# Selected during the tree-based model comparison
# (see notebooks/notebook.ipynb and forecasting/tune_tree_forecast.py).
# max_samples is reduced for the full-data fit so the forest stays within
# memory on a 16 GB machine; it still uses far more rows per tree than the
# tuning run.
FINAL_PARAMS = {
    "n_estimators": 60,
    "max_depth": 14,
    "min_samples_leaf": 20,
    "max_samples": 0.4,
    "max_features": 0.5,
    "n_jobs": 8,
    "random_state": 42,
}


def load_split(path: Path) -> pl.DataFrame:
    """Load the model features and target for one chronological split."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    frame = pl.read_parquet(
        path,
        columns=FEATURES + [TARGET_COLUMN],
    )

    print(f"  {path.name}: {frame.height:,} rows")

    return frame


def main() -> None:
    print("=" * 75)
    print("TRAINING FINAL DEMAND FORECAST MODEL")
    print("=" * 75)

    print("\nLoading training data...")
    train = load_split(TRAIN_PATH)
    validation = load_split(VALIDATION_PATH)

    data = pl.concat([train, validation], how="vertical")
    del train, validation

    n_rows = data.height
    print(f"\nTotal training rows: {n_rows:,}")

    X = np.asarray(
        data.select(FEATURES).to_numpy(),
        dtype=np.float32,
    )
    y = np.asarray(
        data[TARGET_COLUMN].to_numpy(),
        dtype=np.float32,
    )
    del data

    if not np.isfinite(X).all():
        raise ValueError("Training features contain non-finite values.")

    if not np.isfinite(y).all():
        raise ValueError("Training target contains non-finite values.")

    print("\nFitting Random Forest regressor...")
    print(f"  {FINAL_PARAMS}")
    model = RandomForestRegressor(**FINAL_PARAMS)
    model.fit(X, y)
    del X, y

    FORECASTING_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": model,
        "features": FEATURES,
        "target": TARGET_COLUMN,
        "family": "random_forest",
        "params": FINAL_PARAMS,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_rows": n_rows,
    }

    with MODEL_PATH.open("wb") as file:
        pickle.dump(bundle, file)

    print("\n" + "=" * 75)
    print("FINAL MODEL TRAINED")
    print("=" * 75)
    print(f"Model:    {MODEL_PATH}")
    print("Family:   random_forest")
    print(f"Features: {len(FEATURES)}")
    print(f"Rows:     {n_rows:,}")


if __name__ == "__main__":
    main()
