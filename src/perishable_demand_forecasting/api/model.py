"""Load the trained model artifact and generate predictions."""

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from perishable_demand_forecasting.config import FORECASTING_MODELS_DIR

MODEL_PATH = FORECASTING_MODELS_DIR / "final_demand_forecast.joblib"


@lru_cache(maxsize=1)
def load_model(path: Path = MODEL_PATH) -> dict:
    """Load the trained model bundle once per process."""

    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {path}. Run `uv run python train.py` first."
        )

    return joblib.load(path)


def predict_instances(instances: list[dict]) -> list[float]:
    """Validate feature dictionaries and return model predictions."""

    bundle = load_model()
    features = bundle["features"]

    rows = []
    for index, instance in enumerate(instances):
        if not isinstance(instance, dict):
            raise ValueError(f"Instance {index} must be a JSON object.")

        missing = [feature for feature in features if feature not in instance]
        if missing:
            raise ValueError(
                f"Instance {index} is missing features: {', '.join(missing)}"
            )

        try:
            row = [float(instance[feature]) for feature in features]
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Instance {index} contains non-numeric feature values."
            ) from error

        if not np.isfinite(np.asarray(row, dtype=np.float64)).all():
            raise ValueError(f"Instance {index} contains non-finite values.")

        rows.append(row)

    X = np.asarray(rows, dtype=np.float32)
    X = bundle["scaler"].transform(X)
    predictions = bundle["model"].predict(X)

    return [float(value) for value in predictions]
