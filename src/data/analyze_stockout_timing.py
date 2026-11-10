
from pathlib import Path

import polars as pl

DATA_DIR = Path("data/raw/freshretailnet")

FILES = {
    "Train": DATA_DIR / "data/train.parquet",
    "Evaluation": DATA_DIR / "data/eval.parquet",
}

OPERATING_HOUR_START = 6
OPERATING_HOUR_END = 22

GROUP_COLUMNS = ["store_id", "product_id", "dt"]


def load_hourly_data(path: Path) -> pl.LazyFrame:
    """Expand daily arrays into individual operating-hour observations."""
    return (
        pl.scan_parquet(path)
        .select(
            *GROUP_COLUMNS,
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
        .explode(["hour", "sales", "stock_status"])
        .with_columns(
            pl.col("hour").cast(pl.Int8),
            pl.col("sales").cast(pl.Float64),
            pl.col("stock_status").cast(pl.Int8),
        )
        .sort(GROUP_COLUMNS + ["hour"])
    )


def add_episode_information(hourly: pl.LazyFrame) -> pl.LazyFrame:
    """Identify stockout episodes and calculate each hour's position."""
    hourly = hourly.with_columns(
        pl.col("stock_status")
        .shift(1)
        .over(GROUP_COLUMNS)
        .alias("previous_status")
    )

    hourly = hourly.with_columns(
        (
            (pl.col("stock_status") == 1)
            & (
                pl.col("previous_status").is_null()
                | (pl.col("previous_status") == 0)
            )
        )
        .cast(pl.Int8)
        .alias("episode_start")
    )

    hourly = hourly.with_columns(
        pl.col("episode_start")
        .cum_sum()
        .over(GROUP_COLUMNS)
        .alias("episode_id")
    )

    stockout_hours = hourly.filter(
        pl.col("stock_status") == 1
    )

    episode_stats = (
        stockout_hours
        .group_by(GROUP_COLUMNS + ["episode_id"])
        .agg(
            pl.col("hour").min().alias("start_hour"),
            pl.col("hour").max().alias("end_hour"),
            pl.len().alias("duration_hours"),
        )
    )

    return (
        stockout_hours
        .join(
            episode_stats,
            on=GROUP_COLUMNS + ["episode_id"],
            how="left",
        )
        .with_columns(
            pl.when(pl.col("duration_hours") == 1)
            .then(pl.lit("single_hour"))
            .when(pl.col("hour") == pl.col("start_hour"))
            .then(pl.lit("first_hour"))
            .when(pl.col("hour") == pl.col("end_hour"))
            .then(pl.lit("last_hour"))
            .otherwise(pl.lit("middle_hours"))
            .alias("episode_position"),

            pl.when(pl.col("duration_hours") == 1)
            .then(pl.lit("1"))
            .when(pl.col("duration_hours") == 2)
            .then(pl.lit("2"))
            .when(pl.col("duration_hours") <= 5)
            .then(pl.lit("3-5"))
            .when(pl.col("duration_hours") <= 10)
            .then(pl.lit("6-10"))
            .when(pl.col("duration_hours") <= 15)
            .then(pl.lit("11-15"))
            .otherwise(pl.lit("16"))
            .alias("duration_bucket"),
        )
    )


def summarize_episode_positions(stockout_hours: pl.LazyFrame):
    """Summarize sales according to position within stockout episodes."""
    print("\nStockout sales by episode position:")

    position_summary = (
        stockout_hours
        .group_by("episode_position")
        .agg(
            pl.len().alias("hourly_observations"),
            pl.col("sales").mean().alias("mean_sales"),
            pl.col("sales").median().alias("median_sales"),
            (pl.col("sales") > 0).mean().alias("nonzero_sales_rate"),
            pl.col("sales").sum().alias("total_sales"),
        )
        .sort("episode_position")
        .collect()
    )

    print(position_summary)

    print("\nStockout sales by episode duration and position:")

    duration_summary = (
        stockout_hours
        .group_by(["duration_bucket", "episode_position"])
        .agg(
            pl.len().alias("hourly_observations"),
            pl.col("sales").mean().alias("mean_sales"),
            pl.col("sales").median().alias("median_sales"),
            (pl.col("sales") > 0).mean().alias("nonzero_sales_rate"),
        )
        .sort(["duration_bucket", "episode_position"])
        .collect()
    )

    print(duration_summary)


def summarize_transitions(hourly: pl.LazyFrame):
    """Summarize sales according to changes in stock status."""
    print("\nSales by stock-status transition:")

    hourly = hourly.with_columns(
        pl.col("stock_status")
        .shift(1)
        .over(GROUP_COLUMNS)
        .alias("previous_status")
    )

    transitions = (
        hourly
        .with_columns(
            pl.when(pl.col("previous_status").is_null())
            .then(pl.lit("first_hour"))
            .when(
                (pl.col("previous_status") == 0)
                & (pl.col("stock_status") == 0)
            )
            .then(pl.lit("0_to_0"))
            .when(
                (pl.col("previous_status") == 0)
                & (pl.col("stock_status") == 1)
            )
            .then(pl.lit("0_to_1"))
            .when(
                (pl.col("previous_status") == 1)
                & (pl.col("stock_status") == 0)
            )
            .then(pl.lit("1_to_0"))
            .otherwise(pl.lit("1_to_1"))
            .alias("transition")
        )
        .filter(pl.col("transition") != "first_hour")
        .group_by("transition")
        .agg(
            pl.len().alias("observations"),
            pl.col("sales").mean().alias("mean_sales"),
            pl.col("sales").median().alias("median_sales"),
            (pl.col("sales") > 0).mean().alias("nonzero_sales_rate"),
        )
        .sort("transition")
        .collect()
    )

    print(transitions)


def inspect_nonzero_stockout_sales(hourly: pl.LazyFrame):
    """Inspect sales recorded during stockout hours and adjacent hours."""
    print("\nExamples of nonzero sales during stockout hours:")

    contextual = hourly.with_columns(
        pl.col("stock_status")
        .shift(1)
        .over(GROUP_COLUMNS)
        .alias("previous_status"),

        pl.col("sales")
        .shift(1)
        .over(GROUP_COLUMNS)
        .alias("previous_sales"),

        pl.col("stock_status")
        .shift(-1)
        .over(GROUP_COLUMNS)
        .alias("next_status"),

        pl.col("sales")
        .shift(-1)
        .over(GROUP_COLUMNS)
        .alias("next_sales"),
    )

    examples = (
        contextual
        .filter(
            (pl.col("stock_status") == 1)
            & (pl.col("sales") > 0)
        )
        .select(
            *GROUP_COLUMNS,
            "hour",
            "previous_status",
            "previous_sales",
            "sales",
            "next_status",
            "next_sales",
        )
        .collect()
    )

    if examples.is_empty():
        print("No nonzero sales found during stockout hours.")
        return

    print(f"Nonzero stockout-hour observations: {examples.height:,}")

    print(
        examples.sample(
            n=min(30, examples.height),
            seed=42,
        ).sort(GROUP_COLUMNS + ["hour"])
    )


def analyze_dataset(name: str, path: Path):
    print("\n" + "=" * 75)
    print(f"{name.upper()} STOCKOUT TIMING ANALYSIS")
    print("=" * 75)

    hourly = load_hourly_data(path)

    stockout_hours = add_episode_information(hourly)

    summarize_episode_positions(stockout_hours)

    summarize_transitions(hourly)

    inspect_nonzero_stockout_sales(hourly)


def main():
    for name, path in FILES.items():
        if not path.exists():
            print(f"\nDataset not found: {path}")
            continue

        analyze_dataset(name, path)


if __name__ == "__main__":
    main()
