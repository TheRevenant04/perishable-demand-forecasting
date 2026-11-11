
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATA_DIR = Path("data/raw/freshretailnet/data")

FILES = {
    "TRAIN": DATA_DIR / "train.parquet",
    "EVALUATION": DATA_DIR / "eval.parquet",
}

STOCKOUT_COLUMN = "stock_hour6_22_cnt"
SALES_COLUMN = "sale_amount"

# Operating hours are 06:00–21:00.
OPERATING_START_HOUR = 6
OPERATING_END_HOUR = 21

SEVERITY_GROUPS = [
    (0, 0, "Fully in stock", 0),
    (1, 2, "Low exposure", 1),
    (3, 5, "Moderate exposure", 2),
    (6, 9, "High exposure", 3),
    (10, 15, "Very high exposure", 4),
    (16, 16, "Fully out of stock", 5),
]


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_dataset(name: str, path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} dataset not found: {path}")

    print(f"\nLoading {name}: {path}")

    df = pl.read_parquet(path)

    required_columns = [
        SALES_COLUMN,
        STOCKOUT_COLUMN,
        "store_id",
        "product_id",
        "dt",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{name} dataset is missing required columns: "
            f"{missing_columns}"
        )

    return df


# ---------------------------------------------------------------------
# Daily stockout severity
# ---------------------------------------------------------------------

def add_stockout_severity(df: pl.DataFrame) -> pl.DataFrame:
    severity_expression = (
        pl.when(pl.col(STOCKOUT_COLUMN) == 0)
        .then(pl.lit("Fully in stock"))
        .when(pl.col(STOCKOUT_COLUMN) <= 2)
        .then(pl.lit("Low exposure"))
        .when(pl.col(STOCKOUT_COLUMN) <= 5)
        .then(pl.lit("Moderate exposure"))
        .when(pl.col(STOCKOUT_COLUMN) <= 9)
        .then(pl.lit("High exposure"))
        .when(pl.col(STOCKOUT_COLUMN) <= 15)
        .then(pl.lit("Very high exposure"))
        .when(pl.col(STOCKOUT_COLUMN) == 16)
        .then(pl.lit("Fully out of stock"))
        .otherwise(pl.lit("Invalid"))
    )

    order_expression = (
        pl.when(pl.col(STOCKOUT_COLUMN) == 0)
        .then(pl.lit(0))
        .when(pl.col(STOCKOUT_COLUMN) <= 2)
        .then(pl.lit(1))
        .when(pl.col(STOCKOUT_COLUMN) <= 5)
        .then(pl.lit(2))
        .when(pl.col(STOCKOUT_COLUMN) <= 9)
        .then(pl.lit(3))
        .when(pl.col(STOCKOUT_COLUMN) <= 15)
        .then(pl.lit(4))
        .when(pl.col(STOCKOUT_COLUMN) == 16)
        .then(pl.lit(5))
        .otherwise(pl.lit(6))
    )

    return df.with_columns(
        severity_expression.alias("stockout_severity"),
        order_expression.alias("severity_order"),
    )


def analyse_daily_severity(
    name: str,
    df: pl.DataFrame,
) -> None:
    print("\n" + "=" * 75)
    print(f"{name} DAILY STOCKOUT SEVERITY ANALYSIS")
    print("=" * 75)

    df = add_stockout_severity(df)

    # Exclude invalid stockout-hour values from the severity summary.
    valid_df = df.filter(
        pl.col(STOCKOUT_COLUMN).is_between(0, 16)
    )

    summary = (
        valid_df
        .group_by(["severity_order", "stockout_severity"])
        .agg(
            pl.len().alias("observations"),
            pl.col(SALES_COLUMN).mean().alias("avg_daily_sales"),
            pl.col(SALES_COLUMN).median().alias("median_daily_sales"),
            pl.col(SALES_COLUMN).sum().alias("total_daily_sales"),
            (
                (pl.col(SALES_COLUMN) == 0)
                .mean()
                * 100
            ).alias("zero_sales_percentage"),
            pl.col(STOCKOUT_COLUMN).mean().alias("avg_stockout_hours"),
        )
        .sort("severity_order")
        .drop("severity_order")
    )

    print("\nDaily observed sales by stockout severity:")
    print(summary)

    # Compare each severity group against fully in-stock days.
    baseline = (
        valid_df
        .filter(pl.col(STOCKOUT_COLUMN) == 0)
        .select(
            pl.col(SALES_COLUMN)
            .mean()
            .alias("in_stock_avg_daily_sales")
        )
    )

    baseline_value = baseline.item(
        0,
        "in_stock_avg_daily_sales",
    )

    if baseline_value is not None and baseline_value != 0:
        comparison = (
            summary
            .with_columns(
                pl.lit(baseline_value).alias(
                    "in_stock_avg_daily_sales"
                )
            )
            .with_columns(
                (
                    (
                        pl.col("avg_daily_sales")
                        - pl.col("in_stock_avg_daily_sales")
                    )
                    / pl.col("in_stock_avg_daily_sales")
                    * 100
                ).alias("difference_vs_in_stock_percentage")
            )
        )

        print("\nDaily sales comparison against fully in-stock days:")
        print(
            comparison.select(
                "stockout_severity",
                "avg_daily_sales",
                "in_stock_avg_daily_sales",
                "difference_vs_in_stock_percentage",
            )
        )

    # Exact stockout-hour distribution.
    exact_distribution = (
        valid_df
        .group_by(STOCKOUT_COLUMN)
        .agg(
            pl.len().alias("observations"),
            pl.col(SALES_COLUMN).mean().alias("avg_daily_sales"),
            pl.col(SALES_COLUMN).median().alias("median_daily_sales"),
        )
        .sort(STOCKOUT_COLUMN)
    )

    print("\nDaily observations by exact number of stockout hours:")
    print(exact_distribution)

    invalid_count = df.filter(
        ~pl.col(STOCKOUT_COLUMN).is_between(0, 16)
    ).height

    print(f"\nInvalid stockout-hour observations: {invalid_count:,}")
    print(f"Valid daily observations: {valid_df.height:,}")


# ---------------------------------------------------------------------
# Supporting hourly diagnostic
# ---------------------------------------------------------------------

def analyse_sales_during_stockouts(
    name: str,
    df: pl.DataFrame,
) -> None:
    print("\n" + "=" * 75)
    print(f"{name} SUPPORTING HOURLY STOCKOUT DIAGNOSTIC")
    print("=" * 75)

    required_columns = [
        "hours_stock_status",
        "hours_sale",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        print(
            "Skipping hourly diagnostic. Missing columns: "
            f"{missing_columns}"
        )
        return

    # Expand hourly arrays only for diagnostic purposes.
    hourly = (
        df.select(
            "hours_stock_status",
            "hours_sale",
            pl.int_ranges(
                0,
                pl.col("hours_stock_status").list.len(),
            ).alias("hour_index"),
        )
        .explode(
            [
                "hours_stock_status",
                "hours_sale",
                "hour_index",
            ]
        )
        .filter(
            pl.col("hour_index").is_between(
                OPERATING_START_HOUR,
                OPERATING_END_HOUR,
            )
        )
    )

    # Dataset convention:
    # 0 = in stock
    # 1 = out of stock
    status_summary = (
        hourly
        .group_by("hours_stock_status")
        .agg(
            pl.len().alias("hourly_observations"),
            pl.col("hours_sale").sum().alias("total_observed_sales"),
            pl.col("hours_sale").mean().alias("avg_hourly_sales"),
            (
                (pl.col("hours_sale") > 0)
                .mean()
                * 100
            ).alias("nonzero_sales_percentage"),
        )
        .sort("hours_stock_status")
    )

    print("\nHourly sales by stock status (operating hours only):")
    print(status_summary)

    stockout_sales = (
        hourly
        .filter(
            (pl.col("hours_stock_status") == 1)
            & (pl.col("hours_sale") > 0)
        )
        .select(
            pl.len().alias("nonzero_stockout_observations"),
            pl.col("hours_sale")
            .sum()
            .alias("sales_recorded_during_stockouts"),
        )
    )

    print("\nNonzero sales recorded during out-of-stock hours:")
    print(stockout_sales)

    print(
        "\nNote: This is a supporting data diagnostic. "
        "It does not define an hourly demand prediction target."
    )


# ---------------------------------------------------------------------
# Data quality summary
# ---------------------------------------------------------------------

def print_daily_data_quality(
    name: str,
    df: pl.DataFrame,
) -> None:
    print("\n" + "=" * 75)
    print(f"{name} DAILY DATA QUALITY")
    print("=" * 75)

    quality = df.select(
        pl.len().alias("daily_observations"),
        pl.col(STOCKOUT_COLUMN).null_count().alias(
            "missing_stockout_hours"
        ),
        pl.col(SALES_COLUMN).null_count().alias(
            "missing_daily_sales"
        ),
        pl.col(STOCKOUT_COLUMN).min().alias(
            "minimum_stockout_hours"
        ),
        pl.col(STOCKOUT_COLUMN).max().alias(
            "maximum_stockout_hours"
        ),
        pl.col(SALES_COLUMN).min().alias(
            "minimum_daily_sales"
        ),
        pl.col(SALES_COLUMN).max().alias(
            "maximum_daily_sales"
        ),
        pl.col("dt").min().alias("first_date"),
        pl.col("dt").max().alias("last_date"),
    )

    print(quality)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    for name, path in FILES.items():
        df = load_dataset(name, path)

        print_daily_data_quality(name, df)
        analyse_daily_severity(name, df)
        analyse_sales_during_stockouts(name, df)


if __name__ == "__main__":
    main()
