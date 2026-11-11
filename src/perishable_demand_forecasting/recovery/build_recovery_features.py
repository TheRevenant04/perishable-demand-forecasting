
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RECOVERY_DIR = Path("data/processed/freshretailnet/recovery")

OUTPUT_DIR = Path("data/processed/freshretailnet/features")

SPLITS = {
    "train": RECOVERY_DIR / "train.parquet",
    "validation": RECOVERY_DIR / "validation.parquet",
}

TARGET_COLUMN = "target_daily_demand"

DATE_COLUMN = "dt"

IDENTIFIER_COLUMNS = {
    "store_id",
    "product_id",
    "city_id",
    "category_id",
    "group_id",
    "commodity_id",
    "location_id",
    "management_type",
}

EXCLUDED_COLUMNS = {
    "split",
    "target_daily_demand",
    "dt",
}


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_dataset(
    path: Path,
    split_name: str,
) -> pl.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"{split_name} dataset not found: {path}"
        )

    print(f"Loading {split_name}: {path}")

    df = pl.read_parquet(path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Missing target column: {TARGET_COLUMN}"
        )

    if DATE_COLUMN not in df.columns:
        raise ValueError(
            f"Missing date column: {DATE_COLUMN}"
        )

    if df.is_empty():
        raise ValueError(
            f"{split_name} dataset is empty."
        )

    return df


# ---------------------------------------------------------------------
# Calendar features
# ---------------------------------------------------------------------

def add_calendar_features(
    df: pl.DataFrame,
) -> pl.DataFrame:

    date_dtype = df.schema[DATE_COLUMN]

    if date_dtype == pl.String:
        df = df.with_columns(
            pl.col(DATE_COLUMN).str.to_date()
        )

    return df.with_columns(
        pl.col(DATE_COLUMN).dt.weekday().alias("day_of_week"),
        pl.col(DATE_COLUMN).dt.month().alias("month"),
        pl.col(DATE_COLUMN).dt.day().alias("day_of_month"),
        pl.col(DATE_COLUMN).dt.ordinal_day().alias("day_of_year"),
        (
            pl.col(DATE_COLUMN).dt.weekday() >= 6
        ).cast(pl.Int8).alias("is_weekend"),
    )


# ---------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------

def prepare_features(
    df: pl.DataFrame,
    split_name: str,
) -> pl.DataFrame:

    print(f"\nPreparing {split_name} features...")

    df = add_calendar_features(df)

    # Identify available identifier columns.
    categorical_columns = [
        column
        for column in IDENTIFIER_COLUMNS
        if column in df.columns
    ]

    # Convert identifiers to strings so they are treated as
    # categorical variables by the later vectorization step.
    if categorical_columns:
        df = df.with_columns(
            [
                pl.col(column).cast(pl.String)
                for column in categorical_columns
            ]
        )

    # Keep the target separate from the model features.
    # The date is retained for traceability but is not a model feature.
    feature_columns = [
        column
        for column in df.columns
        if column not in EXCLUDED_COLUMNS
    ]

    if not feature_columns:
        raise ValueError(
            f"No features available for {split_name}."
        )

    output_columns = (
        [DATE_COLUMN]
        + feature_columns
        + [TARGET_COLUMN]
    )

    result = df.select(output_columns)

    # Validate target.
    if result[TARGET_COLUMN].null_count() > 0:
        raise ValueError(
            f"{split_name} contains null target values."
        )

    # Validate numerical feature values.
    numeric_columns = [
        column
        for column, dtype in result.schema.items()
        if dtype.is_numeric()
        and column != TARGET_COLUMN
    ]

    for column in numeric_columns:
        invalid_count = result.filter(
            ~pl.col(column).is_finite()
        ).height

        if invalid_count:
            raise ValueError(
                f"{split_name} contains {invalid_count:,} "
                f"non-finite values in feature '{column}'."
            )

    print(f"Rows: {result.height:,}")
    print(f"Total columns: {result.width:,}")
    print(f"Model features: {len(feature_columns):,}")
    print(f"Categorical identifiers: {len(categorical_columns):,}")

    print("\nCategorical identifiers:")
    for column in sorted(categorical_columns):
        print(f"  - {column}")

    return result


# ---------------------------------------------------------------------
# Save datasets
# ---------------------------------------------------------------------

def save_dataset(
    df: pl.DataFrame,
    split_name: str,
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = OUTPUT_DIR / f"{split_name}.parquet"

    df.write_parquet(
        output_path,
        compression="zstd",
    )

    print(f"\nSaved: {output_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    print("=" * 75)
    print("DAILY DEMAND RECOVERY FEATURE PREPARATION")
    print("=" * 75)

    for split_name, input_path in SPLITS.items():

        df = load_dataset(
            input_path,
            split_name,
        )

        prepared_df = prepare_features(
            df,
            split_name,
        )

        save_dataset(
            prepared_df,
            split_name,
        )

    print("\n" + "=" * 75)
    print("FEATURE PREPARATION COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()
