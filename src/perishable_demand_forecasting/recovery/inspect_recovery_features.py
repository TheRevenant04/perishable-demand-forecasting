
from pathlib import Path

import polars as pl

DATA_PATH = Path(
    "data/processed/freshretailnet/features/train.parquet"
)

TARGET_COLUMN = "target_daily_demand"
DATE_COLUMN = "dt"


def main():
    print("=" * 75)
    print("DAILY DEMAND RECOVERY: FEATURE INVENTORY")
    print("=" * 75)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pl.read_parquet(DATA_PATH)

    print(f"\nRows: {df.height:,}")
    print(f"Total columns: {len(df.columns)}")

    print("\nTARGET:")
    if TARGET_COLUMN in df.columns:
        print(f"  {TARGET_COLUMN}: {df.schema[TARGET_COLUMN]}")
        print(f"  Missing values: {df[TARGET_COLUMN].null_count():,}")
    else:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    print("\nDATE:")
    if DATE_COLUMN in df.columns:
        print(f"  {DATE_COLUMN}: {df.schema[DATE_COLUMN]}")
    else:
        print(f"  Warning: date column '{DATE_COLUMN}' not found.")

    print("\nMODEL FEATURES:")
    print("-" * 75)

    excluded = {TARGET_COLUMN, DATE_COLUMN}
    features = [
        column
        for column in df.columns
        if column not in excluded
    ]

    for index, column in enumerate(features, start=1):
        dtype = df.schema[column]
        null_count = df[column].null_count()

        print(
            f"{index:02d}. {column:<40} "
            f"{str(dtype):<15} "
            f"nulls: {null_count:,}"
        )

    print("\n" + "=" * 75)
    print(f"Total model features: {len(features)}")
    print("=" * 75)


if __name__ == "__main__":
    main()
