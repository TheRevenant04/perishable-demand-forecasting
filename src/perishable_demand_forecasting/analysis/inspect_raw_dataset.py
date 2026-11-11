
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATA_DIR = Path("data/raw/freshretailnet/data")

FILES = {
    "Train": DATA_DIR / "train.parquet",
    "Evaluation": DATA_DIR / "eval.parquet",
}

OPERATING_HOUR_START = 6
OPERATING_HOUR_END = 22
OPERATING_HOURS = OPERATING_HOUR_END - OPERATING_HOUR_START

SALES_TOLERANCE = 1e-6

GROUP_COLUMNS = ["store_id", "product_id", "dt"]


# ---------------------------------------------------------------------
# Dataset inspection
# ---------------------------------------------------------------------

def inspect_schema(df: pl.LazyFrame) -> None:
    """Display the dataset schema and dimensions."""

    print("\n1. Dataset schema")

    schema = df.collect_schema()
    print(schema)

    row_count = df.select(pl.len()).collect().item()

    print(f"\nRows: {row_count:,}")
    print(f"Columns: {len(schema):,}")


def inspect_date_coverage(df: pl.LazyFrame) -> None:
    """Report dataset date coverage."""

    print("\n2. Date coverage")

    summary = df.select(
        pl.col("dt").min().alias("start_date"),
        pl.col("dt").max().alias("end_date"),
        pl.col("dt").n_unique().alias("unique_dates"),
    ).collect()

    print(summary)


def inspect_entities(df: pl.LazyFrame) -> None:
    """Report the number of unique entities."""

    print("\n3. Unique entities")

    summary = df.select(
        pl.col("store_id").n_unique().alias("stores"),
        pl.col("product_id").n_unique().alias("products"),
        pl.struct(["store_id", "product_id"])
        .n_unique()
        .alias("store_product_pairs"),
        pl.col("city_id").n_unique().alias("cities"),
    ).collect()

    print(summary)


def inspect_missing_values(df: pl.LazyFrame) -> None:
    """Report null values in columns and hourly arrays."""

    print("\n4. Missing values")

    schema = df.collect_schema()

    missing = df.select(
        [
            pl.col(column).null_count().alias(column)
            for column in schema.names()
        ]
    ).collect()

    print("\nColumn-level null counts:")
    print(missing)

    print("\nNull elements inside hourly arrays:")

    array_missing = df.select(
        pl.col("hours_sale")
        .list.eval(pl.element().is_null())
        .list.sum()
        .sum()
        .alias("null_hourly_sales"),

        pl.col("hours_stock_status")
        .list.eval(pl.element().is_null())
        .list.sum()
        .sum()
        .alias("null_hourly_stock_status"),
    ).collect()

    print(array_missing)


def inspect_sales_distribution(df: pl.LazyFrame) -> None:
    """Report the distribution of daily observed sales."""

    print("\n5. Daily observed sales distribution")

    summary = df.select(
        pl.col("sale_amount").min().alias("minimum"),
        pl.col("sale_amount").max().alias("maximum"),
        pl.col("sale_amount").mean().alias("mean"),
        pl.col("sale_amount").median().alias("median"),
        pl.col("sale_amount").std().alias("std_dev"),
        pl.col("sale_amount").sum().alias("total"),
    ).collect()

    print(summary)


def inspect_stockout_distribution(df: pl.LazyFrame) -> None:
    """Report daily stockout-hour distribution."""

    print("\n6. Daily stockout-hour distribution")

    summary = (
        df.group_by("stock_hour6_22_cnt")
        .agg(
            pl.len().alias("records"),
            pl.col("sale_amount").mean().alias("mean_observed_sales"),
            pl.col("sale_amount").median().alias("median_observed_sales"),
        )
        .sort("stock_hour6_22_cnt")
        .collect()
    )

    print(summary)


def inspect_stock_status(df: pl.LazyFrame) -> None:
    """Report stock-status values during operating hours."""

    print("\n7. Stock-status distribution")

    distribution = (
        df.select(
            pl.col("hours_stock_status")
            .list.slice(
                OPERATING_HOUR_START,
                OPERATING_HOURS,
            )
            .alias("stock_status")
        )
        .explode("stock_status")
        .group_by("stock_status")
        .agg(
            pl.len().alias("observations"),
        )
        .sort("stock_status")
        .collect()
    )

    print(distribution)

    print("\nExpected values:")
    print("0 = In stock")
    print("1 = Out of stock")


def inspect_hourly_arrays(df: pl.LazyFrame) -> None:
    """Validate hourly array lengths."""

    print("\n8. Hourly array lengths")

    summary = df.select(
        pl.col("hours_sale")
        .list.len()
        .min()
        .alias("min_sales_length"),

        pl.col("hours_sale")
        .list.len()
        .max()
        .alias("max_sales_length"),

        pl.col("hours_stock_status")
        .list.len()
        .min()
        .alias("min_stock_length"),

        pl.col("hours_stock_status")
        .list.len()
        .max()
        .alias("max_stock_length"),
    ).collect()

    print(summary)


# ---------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------

def validate_daily_hourly_consistency(
    df: pl.LazyFrame,
) -> None:
    """Compare daily fields against their hourly equivalents."""

    print("\n9. Daily and hourly consistency checks")

    checks = df.select(
        (
            pl.col("hours_sale").list.sum()
            - pl.col("sale_amount")
        )
        .abs()
        .max()
        .alias("max_sales_difference"),

        (
            pl.col("hours_stock_status")
            .list.slice(
                OPERATING_HOUR_START,
                OPERATING_HOURS,
            )
            .list.sum()
            - pl.col("stock_hour6_22_cnt")
        )
        .abs()
        .max()
        .alias("max_stockout_difference"),

        (
            pl.col("stock_hour6_22_cnt")
            .is_between(0, OPERATING_HOURS)
            .not_()
        )
        .sum()
        .alias("invalid_stockout_counts"),

        (
            pl.col("sale_amount") < 0
        )
        .sum()
        .alias("negative_daily_sales"),
    ).collect()

    print(checks)

    max_sales_difference = checks["max_sales_difference"][0]
    max_stockout_difference = checks["max_stockout_difference"][0]

    if (
        max_sales_difference is not None
        and max_sales_difference > SALES_TOLERANCE
    ):
        print("WARNING: Daily sales do not consistently match hourly sales.")

    if (
        max_stockout_difference is not None
        and max_stockout_difference != 0
    ):
        print("WARNING: Daily stockout counts do not match hourly status.")

    if checks["invalid_stockout_counts"][0] > 0:
        print("WARNING: Invalid daily stockout-hour counts detected.")

    if checks["negative_daily_sales"][0] > 0:
        print("WARNING: Negative daily sales detected.")


def validate_stock_status_values(df: pl.LazyFrame) -> None:
    """Check that operating-hour stock statuses are binary."""

    print("\n10. Invalid stock-status values")

    invalid = (
        df.select(
            pl.col("hours_stock_status")
            .list.slice(
                OPERATING_HOUR_START,
                OPERATING_HOURS,
            )
            .alias("stock_status")
        )
        .explode("stock_status")
        .filter(
            ~pl.col("stock_status").is_in([0, 1])
        )
        .select(pl.len().alias("invalid_observations"))
        .collect()
    )

    invalid_count = invalid["invalid_observations"][0]

    print(f"Invalid operating-hour observations: {invalid_count:,}")


def validate_duplicate_records(df: pl.LazyFrame) -> None:
    """Check for duplicate store-product-date combinations."""

    print("\n11. Duplicate records")

    duplicates = (
        df.group_by(GROUP_COLUMNS)
        .len()
        .filter(pl.col("len") > 1)
        .select(
            pl.len().alias("duplicate_groups"),
        )
        .collect()
    )

    duplicate_count = duplicates["duplicate_groups"][0]

    print(f"Duplicate store-product-date groups: {duplicate_count:,}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def inspect_file(name: str, path: Path) -> None:
    """Run the complete inspection and validation workflow."""

    print("\n" + "=" * 75)
    print(f"{name.upper()} DATASET INSPECTION")
    print("=" * 75)

    if not path.exists():
        print(f"Dataset not found: {path}")
        return

    df = pl.scan_parquet(path)

    inspect_schema(df)
    inspect_date_coverage(df)
    inspect_entities(df)
    inspect_missing_values(df)
    inspect_sales_distribution(df)
    inspect_stockout_distribution(df)
    inspect_stock_status(df)
    inspect_hourly_arrays(df)

    validate_daily_hourly_consistency(df)
    validate_stock_status_values(df)
    validate_duplicate_records(df)


def main() -> None:
    for name, path in FILES.items():
        inspect_file(name, path)


if __name__ == "__main__":
    main()
