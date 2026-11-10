
from pathlib import Path

import polars as pl

DATA_DIR = Path("data/raw/freshretailnet/data")

DATASETS = {
    "TRAIN": DATA_DIR / "train.parquet",
    "EVALUATION": DATA_DIR / "eval.parquet",
}

GROUP_KEYS = ["store_id", "product_id", "dt"]

OPERATING_START_HOUR = 6
OPERATING_END_HOUR = 22
OPERATING_HOURS = OPERATING_END_HOUR - OPERATING_START_HOUR


def load_hourly_data(path: Path) -> pl.DataFrame:
    """
    Load daily records and expand hourly sales and stock status
    into individual store-product-hour observations.
    """

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    print(f"\nLoading: {path}")

    hourly = (
        pl.scan_parquet(path)
        .select(
            "store_id",
            "product_id",
            "dt",
            "hours_sale",
            "hours_stock_status",
        )
        .with_columns(
            pl.col("hours_sale")
            .list.slice(OPERATING_START_HOUR, OPERATING_HOURS)
            .alias("sales"),
            pl.col("hours_stock_status")
            .list.slice(OPERATING_START_HOUR, OPERATING_HOURS)
            .alias("stock_status"),
            pl.int_ranges(
                OPERATING_START_HOUR,
                OPERATING_END_HOUR,
            ).alias("hour"),
        )
        .explode(["sales", "stock_status", "hour"])
        .rename({"sales": "sales"})
        .with_columns(
            pl.col("sales").cast(pl.Float64),
            pl.col("stock_status").cast(pl.Int8),
            pl.col("hour").cast(pl.Int8),
        )
        .collect()
    )

    print(f"Hourly observations: {hourly.height:,}")

    return hourly


def classify_stockout_positions(
    hourly: pl.DataFrame,
) -> pl.DataFrame:
    """
    Identify the position of each stockout observation
    within its continuous stockout episode.

    Positions:
    - first_hour
    - middle_hours
    - last_hour
    - single_hour
    """

    hourly = hourly.sort(
        GROUP_KEYS + ["hour"]
    )

    hourly = hourly.with_columns(
        pl.col("stock_status")
        .shift(1)
        .over(GROUP_KEYS)
        .alias("previous_status"),

        pl.col("stock_status")
        .shift(-1)
        .over(GROUP_KEYS)
        .alias("next_status"),
    )

    hourly = hourly.with_columns(
        (
            (pl.col("stock_status") == 1)
            & (
                pl.col("previous_status").is_null()
                | (pl.col("previous_status") == 0)
            )
        ).alias("is_first_hour"),

        (
            (pl.col("stock_status") == 1)
            & (
                pl.col("next_status").is_null()
                | (pl.col("next_status") == 0)
            )
        ).alias("is_last_hour"),
    )

    hourly = hourly.with_columns(
        pl.when(pl.col("stock_status") == 0)
        .then(pl.lit("in_stock"))
        .when(
            pl.col("is_first_hour")
            & pl.col("is_last_hour")
        )
        .then(pl.lit("single_hour"))
        .when(pl.col("is_first_hour"))
        .then(pl.lit("first_hour"))
        .when(pl.col("is_last_hour"))
        .then(pl.lit("last_hour"))
        .otherwise(pl.lit("middle_hours"))
        .alias("episode_position")
    )

    return hourly


def print_overall_summary(
    hourly: pl.DataFrame,
    dataset_name: str,
) -> None:
    """
    Compare observed sales with sales retained after
    strict stockout masking.
    """

    print("\n" + "=" * 75)
    print(f"{dataset_name} STOCKOUT MASKING ANALYSIS")
    print("=" * 75)

    summary = hourly.select(
        pl.len().alias("total_observations"),

        pl.col("sales").sum().alias("raw_sales"),

        pl.col("sales")
        .filter(pl.col("stock_status") == 0)
        .sum()
        .alias("retained_sales"),

        pl.col("sales")
        .filter(pl.col("stock_status") == 1)
        .sum()
        .alias("discarded_sales"),

        (pl.col("stock_status") == 1)
        .sum()
        .alias("stockout_observations"),

        (
            (pl.col("stock_status") == 1)
            & (pl.col("sales") > 0)
        )
        .sum()
        .alias("nonzero_stockout_observations"),

        (
            (pl.col("stock_status") == 1)
            & (pl.col("sales") == 0)
        )
        .sum()
        .alias("zero_sales_stockout_observations"),
    ).row(0, named=True)

    raw_sales = summary["raw_sales"] or 0.0
    retained_sales = summary["retained_sales"] or 0.0
    discarded_sales = summary["discarded_sales"] or 0.0

    print(f"\nTotal hourly observations: "
          f"{summary['total_observations']:,}")

    print(f"Stockout observations: "
          f"{summary['stockout_observations']:,}")

    print(
        f"Nonzero sales during stockouts: "
        f"{summary['nonzero_stockout_observations']:,}"
    )

    print(
        f"Zero-sales observations during stockouts: "
        f"{summary['zero_sales_stockout_observations']:,}"
    )

    print("\nSales comparison")
    print("-" * 50)

    print(f"Raw observed sales:       {raw_sales:,.6f}")
    print(f"Sales retained:           {retained_sales:,.6f}")
    print(f"Sales discarded:          {discarded_sales:,.6f}")

    print("\nMasking impact")
    print("-" * 50)

    discarded_pct = (
        discarded_sales / raw_sales * 100
        if raw_sales > 0
        else 0.0
    )

    retained_pct = (
        retained_sales / raw_sales * 100
        if raw_sales > 0
        else 0.0
    )

    nonzero_stockout_pct = (
        summary["nonzero_stockout_observations"]
        / summary["stockout_observations"]
        * 100
        if summary["stockout_observations"] > 0
        else 0.0
    )

    print(f"Percentage of sales discarded: {discarded_pct:.4f}%")
    print(f"Percentage of sales retained:  {retained_pct:.4f}%")
    print(
        f"Stockout observations with sales: "
        f"{nonzero_stockout_pct:.4f}%"
    )

    print("\nStrict masking interpretation:")
    print(
        f"  {discarded_pct:.2f}% of recorded sales would be "
        "excluded from the observed-demand series."
    )


def print_position_summary(
    hourly: pl.DataFrame,
    dataset_name: str,
) -> None:
    """
    Quantify sales discarded by strict masking,
    broken down by stockout episode position.
    """

    print("\n" + "=" * 75)
    print(f"{dataset_name} SALES BY STOCKOUT EPISODE POSITION")
    print("=" * 75)

    stockout_data = hourly.filter(
        pl.col("stock_status") == 1
    )

    summary = (
        stockout_data
        .group_by("episode_position")
        .agg(
            pl.len().alias("observations"),

            pl.col("sales")
            .sum()
            .alias("discarded_sales"),

            pl.col("sales")
            .mean()
            .alias("mean_sales"),

            pl.col("sales")
            .median()
            .alias("median_sales"),

            (pl.col("sales") > 0)
            .sum()
            .alias("nonzero_observations"),

            (pl.col("sales") == 0)
            .sum()
            .alias("zero_observations"),
        )
        .sort("episode_position")
    )

    print(summary)

    print("\nPosition-level percentages")
    print("-" * 50)

    total_discarded = (
        stockout_data
        .select(pl.col("sales").sum())
        .item()
        or 0.0
    )

    total_observations = stockout_data.height

    for row in summary.iter_rows(named=True):
        position = row["episode_position"]

        sales = row["discarded_sales"] or 0.0

        sales_pct = (
            sales / total_discarded * 100
            if total_discarded > 0
            else 0.0
        )

        observation_pct = (
            row["observations"] / total_observations * 100
            if total_observations > 0
            else 0.0
        )

        nonzero_pct = (
            row["nonzero_observations"]
            / row["observations"]
            * 100
            if row["observations"] > 0
            else 0.0
        )

        print(f"\n{position}")
        print(f"  Observations:             {row['observations']:,}")
        print(f"  Share of stockout hours:  {observation_pct:.2f}%")
        print(f"  Discarded sales:          {sales:,.6f}")
        print(f"  Share of discarded sales: {sales_pct:.2f}%")
        print(f"  Mean sales:               {row['mean_sales']:.6f}")
        print(f"  Median sales:             {row['median_sales']:.6f}")
        print(f"  Nonzero sales rate:       {nonzero_pct:.4f}%")


def create_masked_sales(
    hourly: pl.DataFrame,
) -> pl.DataFrame:
    """
    Create an explicitly masked sales column.

    In-stock observations retain their sales.
    Stockout observations become null, not zero.
    """

    return hourly.with_columns(
        pl.when(pl.col("stock_status") == 0)
        .then(pl.col("sales"))
        .otherwise(None)
        .alias("masked_sales")
    )


def analyze_dataset(
    dataset_name: str,
    path: Path,
) -> None:

    hourly = load_hourly_data(path)

    print_overall_summary(
        hourly,
        dataset_name,
    )

    hourly = classify_stockout_positions(hourly)

    print_position_summary(
        hourly,
        dataset_name,
    )

    masked = create_masked_sales(hourly)

    print("\n" + "=" * 75)
    print(f"{dataset_name} MASKED SALES VALIDATION")
    print("=" * 75)

    validation = masked.select(
        pl.col("sales").sum().alias("raw_sales"),
        pl.col("masked_sales").sum().alias("masked_sales"),
        pl.col("masked_sales").is_null().sum().alias("null_observations"),
        pl.col("stock_status").filter(
            pl.col("masked_sales").is_null()
        ).eq(1).all().alias("all_nulls_are_stockouts"),
    )

    print(validation)


def main() -> None:

    for dataset_name, path in DATASETS.items():
        analyze_dataset(
            dataset_name,
            path,
        )

    print("\n" + "=" * 75)
    print("STOCKOUT MASKING ANALYSIS COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()
