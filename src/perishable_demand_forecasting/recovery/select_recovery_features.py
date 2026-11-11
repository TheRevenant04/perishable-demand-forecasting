
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

INPUT_DIR = Path("data/processed/freshretailnet/features")
OUTPUT_DIR = INPUT_DIR / "experiments"

SPLITS = ("train", "validation")

DATE_COLUMN = "dt"
TARGET_COLUMN = "target_daily_demand"

CATEGORICAL_COLUMNS = [
    "city_id",
    "store_id",
    "product_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
    "management_group_id",
]


# ---------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------

CORE_FEATURES = [
    "simulated_observed_daily_sales",
    "simulated_stockout_hours",
    "city_id",
    "store_id",
    "product_id",
    "day_of_week",
    "month",
]

PRODUCT_FEATURES = [
    "first_category_id",
    "second_category_id",
    "third_category_id",
]

BUSINESS_FEATURES = [
    "discount",
    "holiday_flag",
    "activity_flag",
]

WEATHER_FEATURES = [
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level",
]

HOURLY_FEATURES = [
    f"observed_hour_{hour:02d}"
    for hour in range(6, 22)
]

MANAGEMENT_FEATURES = [
    "management_group_id",
]

ADDITIONAL_DEMAND_FEATURES = [
    "simulated_stockout_fraction",
]

ADDITIONAL_HOURLY_FEATURES = [
    f"observed_hour_{hour:02d}"
    for hour in list(range(0, 6)) + [22, 23]
]

ADDITIONAL_CALENDAR_FEATURES = [
    "day_of_month",
    "day_of_year",
    "is_weekend",
]


# ---------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------

EXPERIMENTS = {
    "A_core": CORE_FEATURES,

    "B_product": (
        CORE_FEATURES
        + PRODUCT_FEATURES
    ),

    "C_business": (
        CORE_FEATURES
        + BUSINESS_FEATURES
    ),

    "D_weather": (
        CORE_FEATURES
        + WEATHER_FEATURES
    ),

    "E_hourly": (
        CORE_FEATURES
        + HOURLY_FEATURES
    ),

    "F_management": (
        CORE_FEATURES
        + MANAGEMENT_FEATURES
    ),

    "G_business_hourly": (
        CORE_FEATURES
        + BUSINESS_FEATURES
        + HOURLY_FEATURES
    ),

    "H_all_features": list(dict.fromkeys(
        CORE_FEATURES
        + PRODUCT_FEATURES
        + BUSINESS_FEATURES
        + WEATHER_FEATURES
        + HOURLY_FEATURES
        + MANAGEMENT_FEATURES
        + ADDITIONAL_DEMAND_FEATURES
        + ADDITIONAL_HOURLY_FEATURES
        + ADDITIONAL_CALENDAR_FEATURES
    )),
}


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_dataset(split: str) -> pl.DataFrame:
    path = INPUT_DIR / f"{split}.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"{split} dataset not found: {path}"
        )

    print(f"Loading {split}: {path}")

    return pl.read_parquet(path)


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_columns(
    df: pl.DataFrame,
    split: str,
    experiment: str,
    features: list[str],
) -> None:
    required = [
        DATE_COLUMN,
        TARGET_COLUMN,
        *features,
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{split}/{experiment}: "
            f"Missing columns: {missing}"
        )


def prepare_experiment(
    df: pl.DataFrame,
    split: str,
    experiment: str,
    features: list[str],
) -> pl.DataFrame:
    validate_columns(
        df,
        split,
        experiment,
        features,
    )

    columns = [
        DATE_COLUMN,
        *features,
        TARGET_COLUMN,
    ]

    result = df.select(columns)

    categorical = [
        column
        for column in CATEGORICAL_COLUMNS
        if column in result.columns
    ]

    if categorical:
        result = result.with_columns(
            [
                pl.col(column).cast(pl.String)
                for column in categorical
            ]
        )

    if result[TARGET_COLUMN].null_count() > 0:
        raise ValueError(
            f"{split}/{experiment}: Target contains nulls."
        )

    if result.filter(
        ~pl.col(TARGET_COLUMN).is_finite()
    ).height > 0:
        raise ValueError(
            f"{split}/{experiment}: Target contains "
            "non-finite values."
        )

    for column in features:
        if result[column].null_count() > 0:
            raise ValueError(
                f"{split}/{experiment}: "
                f"{column} contains nulls."
            )

        if result.schema[column].is_numeric():
            if result.filter(
                ~pl.col(column).is_finite()
            ).height > 0:
                raise ValueError(
                    f"{split}/{experiment}: "
                    f"{column} contains non-finite values."
                )

    return result


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("=" * 75)
    print("DAILY DEMAND RECOVERY: CONTROLLED FEATURE EXPERIMENTS")
    print("=" * 75)

    datasets = {
        split: load_dataset(split)
        for split in SPLITS
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for experiment, features in EXPERIMENTS.items():
        print("\n" + "=" * 75)
        print(f"EXPERIMENT: {experiment}")
        print("=" * 75)

        print(f"Selected features: {len(features)}")

        experiment_dir = OUTPUT_DIR / experiment

        experiment_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for split, df in datasets.items():
            print(f"\nPreparing {split}...")

            result = prepare_experiment(
                df=df,
                split=split,
                experiment=experiment,
                features=features,
            )

            output_path = (
                experiment_dir / f"{split}.parquet"
            )

            result.write_parquet(output_path)

            print(f"Rows: {result.height:,}")
            print(f"Columns: {len(result.columns)}")
            print(f"Saved: {output_path}")

    print("\n" + "=" * 75)
    print("CONTROLLED FEATURE EXPERIMENTS COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()
