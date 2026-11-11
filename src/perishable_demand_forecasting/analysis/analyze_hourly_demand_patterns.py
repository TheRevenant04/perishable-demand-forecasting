
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
OPERATING_HOURS = range(
    OPERATING_HOUR_START,
    OPERATING_HOUR_END,
)


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------

def load_hourly_data(path: Path) -> pl.LazyFrame:
    """Expand daily hourly arrays into individual hourly observations."""

    return (
        pl.scan_parquet(path)
        .select(
            "store_id",
            "product_id",
            "dt",
            "hours_sale",
            "hours_stock_status",
        )
        .with_columns(
            pl.int_ranges(0, 24).alias("hour"),
        )
        .explode(
            [
                "hour",
                "hours_sale",
                "hours_stock_status",
            ]
        )
        .rename(
            {
                "hours_sale": "hourly_sales",
                "hours_stock_status": "stock_status",
            }
        )
        .filter(
            pl.col("hour").is_between(
                OPERATING_HOUR_START,
                OPERATING_HOUR_END - 1,
            )
        )
        .with_columns(
            pl.col("hour").cast(pl.Int8),
        )
    )


# ---------------------------------------------------------------------
# Hourly demand profile
# ---------------------------------------------------------------------

def analyse_overall_hourly_profile(
    hourly: pl.LazyFrame,
) -> None:
    """Summarise observed sales by hour across all stock statuses."""

    print("\n1. Overall hourly observed-sales profile")

    summary = (
        hourly.group_by("hour")
        .agg(
            pl.len().alias("observations"),
            pl.col("hourly_sales").sum().alias("total_observed_sales"),
            pl.col("hourly_sales").mean().alias("mean_observed_sales"),
            pl.col("hourly_sales").median().alias("median_observed_sales"),
            (pl.col("hourly_sales") == 0)
            .mean()
            .alias("zero_sales_percentage"),
        )
        .sort("hour")
        .collect()
    )

    print(summary)


def analyse_hourly_sales_by_stock_status(
    hourly: pl.LazyFrame,
) -> None:
    """Compare observed sales for in-stock and out-of-stock observations."""

    print("\n2. Hourly observed sales by stock status")

    summary = (
        hourly.group_by(
            [
                "hour",
                "stock_status",
            ]
        )
        .agg(
            pl.len().alias("observations"),
            pl.col("hourly_sales").sum().alias("total_observed_sales"),
            pl.col("hourly_sales").mean().alias("mean_observed_sales"),
            pl.col("hourly_sales").median().alias("median_observed_sales"),
            (pl.col("hourly_sales") == 0)
            .mean()
            .alias("zero_sales_percentage"),
            (pl.col("hourly_sales") > 0)
            .mean()
            .alias("nonzero_sales_percentage"),
        )
        .sort(
            [
                "hour",
                "stock_status",
            ]
        )
        .collect()
    )

    print(summary)


def analyse_stockout_frequency_by_hour(
    hourly: pl.LazyFrame,
) -> None:
    """Calculate the proportion of observations marked out of stock by hour."""

    print("\n3. Stockout frequency by hour")

    summary = (
        hourly.group_by("hour")
        .agg(
            pl.len().alias("total_observations"),
            (pl.col("stock_status") == 1)
            .sum()
            .alias("out_of_stock_observations"),
            (pl.col("stock_status") == 0)
            .sum()
            .alias("in_stock_observations"),
            (pl.col("stock_status") == 1)
            .mean()
            .alias("stockout_percentage"),
        )
        .sort("hour")
        .collect()
    )

    print(summary)


def analyse_hourly_sales_distribution(
    hourly: pl.LazyFrame,
) -> None:
    """Inspect sales distribution by hour and stock status."""

    print("\n4. Hourly sales distribution")

    summary = (
        hourly.group_by(
            [
                "hour",
                "stock_status",
            ]
        )
        .agg(
            pl.col("hourly_sales").min().alias("minimum_sales"),
            pl.col("hourly_sales").quantile(0.25).alias("q25_sales"),
            pl.col("hourly_sales").median().alias("median_sales"),
            pl.col("hourly_sales").quantile(0.75).alias("q75_sales"),
            pl.col("hourly_sales").max().alias("maximum_sales"),
        )
        .sort(
            [
                "hour",
                "stock_status",
            ]
        )
        .collect()
    )

    print(summary)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def analyse_file(name: str, path: Path) -> None:
    """Run hourly demand analysis for one dataset."""

    print("\n" + "=" * 75)
    print(f"{name.upper()} HOURLY DEMAND PROFILE")
    print("=" * 75)

    if not path.exists():
        print(f"Dataset not found: {path}")
        return

    hourly = load_hourly_data(path)

    analyse_overall_hourly_profile(hourly)
    analyse_hourly_sales_by_stock_status(hourly)
    analyse_stockout_frequency_by_hour(hourly)
    analyse_hourly_sales_distribution(hourly)


def main() -> None:
    for name, path in FILES.items():
        analyse_file(name, path)


if __name__ == "__main__":
    main()
