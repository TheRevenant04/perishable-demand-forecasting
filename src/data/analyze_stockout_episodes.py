
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
    """Extract hourly sales and stock availability."""

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
            pl.col("stock_status").cast(pl.Int8),
            pl.col("sales").cast(pl.Float64),
        )
        .sort(GROUP_COLUMNS + ["hour"])
    )


def identify_stockout_episodes(
    hourly: pl.LazyFrame,
) -> pl.LazyFrame:
    """Identify consecutive stockout hours within each day."""

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

    return (
        hourly
        .filter(pl.col("stock_status") == 1)
        .group_by(GROUP_COLUMNS + ["episode_id"])
        .agg(
            pl.col("hour").min().alias("start_hour"),
            pl.col("hour").max().alias("end_hour"),
            pl.len().alias("duration_hours"),
            pl.col("sales").sum().alias("episode_sales"),
            pl.col("sales").mean().alias("mean_hourly_sales"),
            (pl.col("sales") > 0)
            .sum()
            .alias("nonzero_sales_hours"),
        )
    )


def attach_episode_context(
    episodes: pl.DataFrame,
    hourly: pl.LazyFrame,
) -> pl.DataFrame:
    """Attach sales immediately before and after each episode."""

    hourly_df = hourly.select(
        *GROUP_COLUMNS,
        "hour",
        "sales",
    ).collect()

    before = hourly_df.rename(
        {
            "hour": "before_hour",
            "sales": "sales_before",
        }
    )

    after = hourly_df.rename(
        {
            "hour": "after_hour",
            "sales": "sales_after",
        }
    )

    episodes = episodes.with_columns(
        (pl.col("start_hour") - 1).alias("before_hour"),
        (pl.col("end_hour") + 1).alias("after_hour"),
    )

    episodes = (
        episodes
        .join(
            before,
            on=GROUP_COLUMNS + ["before_hour"],
            how="left",
        )
        .join(
            after,
            on=GROUP_COLUMNS + ["after_hour"],
            how="left",
        )
        .with_columns(
            pl.when(pl.col("before_hour") < OPERATING_HOUR_START)
            .then(None)
            .otherwise(pl.col("sales_before"))
            .alias("sales_before"),

            pl.when(pl.col("after_hour") >= OPERATING_HOUR_END)
            .then(None)
            .otherwise(pl.col("sales_after"))
            .alias("sales_after"),
        )
    )

    return episodes


def summarize_episodes(
    episodes: pl.DataFrame,
    name: str,
):
    """Generate episode duration and boundary summaries."""

    print("\n" + "=" * 75)
    print(f"{name.upper()} STOCKOUT EPISODE ANALYSIS")
    print("=" * 75)

    print(f"\nTotal stockout episodes: {episodes.height:,}")

    print("\nEpisode duration distribution:")

    duration_summary = (
        episodes
        .group_by("duration_hours")
        .agg(
            pl.len().alias("episodes"),
            pl.col("episode_sales")
            .mean()
            .alias("mean_episode_sales"),
            pl.col("episode_sales")
            .median()
            .alias("median_episode_sales"),
            pl.col("episode_sales")
            .sum()
            .alias("total_episode_sales"),
            (pl.col("episode_sales") > 0)
            .mean()
            .alias("episodes_with_sales_rate"),
            pl.col("nonzero_sales_hours")
            .mean()
            .alias("mean_nonzero_sales_hours"),
        )
        .sort("duration_hours")
    )

    print(duration_summary)

    print("\nOverall episode statistics:")

    overall_summary = episodes.select(
        pl.len().alias("total_episodes"),
        pl.col("duration_hours").mean().alias("mean_duration"),
        pl.col("duration_hours").median().alias("median_duration"),
        pl.col("duration_hours").max().alias("maximum_duration"),
        pl.col("episode_sales").mean().alias("mean_episode_sales"),
        pl.col("episode_sales").median().alias("median_episode_sales"),
        (pl.col("episode_sales") > 0)
        .mean()
        .alias("episodes_with_sales_rate"),
        pl.col("sales_before").mean().alias("mean_sales_before"),
        pl.col("sales_after").mean().alias("mean_sales_after"),
    )

    print(overall_summary)

    print("\nEpisode boundary sales:")

    boundary_summary = episodes.select(
        pl.col("sales_before")
        .mean()
        .alias("mean_sales_before"),

        pl.col("sales_after")
        .mean()
        .alias("mean_sales_after"),

        (pl.col("sales_before") > 0)
        .mean()
        .alias("nonzero_sales_before_rate"),

        (pl.col("sales_after") > 0)
        .mean()
        .alias("nonzero_sales_after_rate"),
    )

    print(boundary_summary)


def analyze_dataset(name: str, path: Path):
    print(f"\nProcessing dataset: {path}")

    hourly = load_hourly_data(path)

    episodes = identify_stockout_episodes(hourly).collect()

    episodes = attach_episode_context(episodes, hourly)

    summarize_episodes(episodes, name)


def main():
    for name, path in FILES.items():
        if not path.exists():
            print(f"\nDataset not found: {path}")
            continue

        analyze_dataset(name, path)


if __name__ == "__main__":
    main()
