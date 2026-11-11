"""Central project paths and constants.

Paths are resolved relative to the repository root so scripts can be run
from anywhere (e.g. ``python -m perishable_demand_forecasting.recovery.train_recovery_experiments``).
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    """Return an environment override for a project path when provided."""

    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


DATA_DIR = _env_path("PERISHABLE_DATA_DIR", PROJECT_ROOT / "data")
RAW_DATA_DIR = DATA_DIR / "raw" / "freshretailnet" / "data"
PROCESSED_DIR = DATA_DIR / "processed" / "freshretailnet"

FEATURES_DIR = PROCESSED_DIR / "features"
EXPERIMENTS_DIR = FEATURES_DIR / "experiments"
RECOVERY_DIR = PROCESSED_DIR / "recovery"
FORECASTING_DIR = PROCESSED_DIR / "forecasting"

MODELS_DIR = _env_path("PERISHABLE_MODELS_DIR", PROJECT_ROOT / "models")
RECOVERY_MODELS_DIR = MODELS_DIR / "recovery"
FORECASTING_MODELS_DIR = MODELS_DIR / "forecasting"

TRAIN_FILE = RAW_DATA_DIR / "train.parquet"
EVAL_FILE = RAW_DATA_DIR / "eval.parquet"

DATASET_ID = "Dingdong-Inc/FreshRetailNet-50K"
