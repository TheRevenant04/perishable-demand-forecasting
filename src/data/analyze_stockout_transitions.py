
from pathlib import Path

import polars as pl

DATA_DIR = Path("data/raw/freshretailnet")

FILES = {
    "Train": DATA_DIR / "data/train.parquet",
    "Evaluation": DATA_DIR / "data/eval.parquet",
}

OPERATING_HOUR_START = 6
OPERATING_HOUR_END = 22


def load_hourly_data(path: Path) -> pl.LazyFrame:
    """
    Extract hourly sales and stock availability for operating hours.

    Stock status:
        0 = In stock
        1 = Out of stock

    Operating hours:
        06:00 to 21:00
    """

    df = pl.scan_parquet(path)

    hourly = (
        df.select(
            "store_id",
            "product_id",
            "dt",
            pl.col("hours_sale")
            .list.slice(OPERATING_HOUR_START, 16)
            .alias("sales"),
            pl.col("hours_stock_status")
            .list.slice(OPERATING_HOUR_START, 16)
            .alias("stock_status"),
        )
        .with_columns(
            pl.int_ranges(
                OPERATING_HOUR_START,
                OPERATING_HOUR_END,
            ).alias("hour")
        )
        .explode(
            ["hour", "sales", "stock_status"]
        )
        .with_columns(
            pl.col("stock_status").cast(pl.Int8),
            pl.col("sales").cast(pl.Float64),
        )
        .sort(
            ["store_id", "product_id", "dt", "hour"]
        )
    )

    return hourly


def create_transitions(hourly: pl.LazyFrame) -> pl.LazyFrame:
    """
    Identify stock-status transitions between consecutive hours
    within each store-product-day combination.
    """

    transitions = (
        hourly
        .with_columns(
            pl.col("stock_status")
            .shift(1)
            .over(["store_id", "product_id", "dt"])
            .alias("previous_status"),

            pl.col("sales")
            .shift(1)
            .over(["store_id", "product_id", "dt"])
            .alias("previous_sales"),
        )
        .filter(
            pl.col("previous_status").is_not_null()
        )
        .with_columns(
            pl.concat_str(
                [
                    pl.col("previous_status").cast(pl.String),
                    pl.lit(" -> "),
                    pl.col("stock_status").cast(pl.String),
                ]
            ).alias("transition")
        )
    )

    return transitions


def summarize_transitions(
    transitions: pl.LazyFrame,
) -> pl.DataFrame:
    """
    Summarize sales behaviour for each stock-status transition.
    """

    summary = (
        transitions
        .group_by("transition")
        .agg(
            pl.len().alias("observations"),

            pl.col("sales")
            .mean()
            .alias("current_hour_mean_sales"),

            pl.col("sales")
            .median()
            .alias("current_hour_median_sales"),

            pl.col("sales")
            .sum()
            .alias("current_hour_total_sales"),

            (pl.col("sales") > 0)
            .mean()
            .alias("current_hour_nonzero_rate"),

            pl.col("previous_sales")
            .mean()
            .alias("previous_hour_mean_sales"),

            (pl.col("previous_sales") > 0)
            .mean()
            .alias("previous_hour_nonzero_rate"),
        )
        .sort("transition")
        .collect()
    )

    return summary


def summarize_hourly_transitions(
    transitions: pl.LazyFrame,
) -> pl.DataFrame:
    """
    Summarize transition behaviour separately for each current hour.
    """

    hourly_summary = (
        transitions
        .group_by(["hour", "transition"])
        .agg(
            pl.len().alias("observations"),

            pl.col("sales")
            .mean()
            .alias("mean_sales"),

            (pl.col("sales") > 0)
            .mean()
            .alias("nonzero_sales_rate"),
        )
        .sort(["hour", "transition"])
        .collect()
    )

    return hourly_summary


def analyze_transitions(name: str, path: Path):
    """
    Run the complete stockout transition analysis for a dataset.
    """

    print("\n" + "=" * 75)
    print(f"{name.upper()} STOCKOUT TRANSITION ANALYSIS")
    print("=" * 75)

    print(f"\nDataset: {path}")

    hourly = load_hourly_data(path)

    transitions = create_transitions(hourly)

    print("\nTransition summary:")

    summary = summarize_transitions(transitions)

    print(summary)

    print("\nTransition counts by current hour:")

    hourly_summary = summarize_hourly_transitions(transitions)

    print(hourly_summary)


def main():
    for name, path in FILES.items():

        if not path.exists():
            print(f"\nDataset not found: {path}")
            continue

        analyze_transitions(name, path)


if __name__ == "__main__":
    main()
