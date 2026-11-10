
from pathlib import Path

import polars as pl

DATA_DIR = Path("data/raw/freshretailnet")

FILES = {
    "Train": DATA_DIR / "data/train.parquet",
    "Evaluation": DATA_DIR / "data/eval.parquet",
}


def analyze_stockouts(name: str, path: Path):
    print(f"\n{'=' * 60}")
    print(f"{name} STOCKOUT ANALYSIS")
    print(f"{'=' * 60}")

    df = pl.scan_parquet(path)

    # 1. Daily sales by stockout severity
    print("\n1. Daily sales by number of stockout hours")

    daily_analysis = (
        df.group_by("stock_hour6_22_cnt")
        .agg(
            pl.len().alias("records"),
            pl.col("sale_amount").mean().alias("mean_sales"),
            pl.col("sale_amount").median().alias("median_sales"),
            pl.col("sale_amount").sum().alias("total_sales"),
        )
        .sort("stock_hour6_22_cnt")
        .collect()
    )

    print(daily_analysis)

    # 2. Hourly sales by stock availability
    print("\n2. Hourly sales by stock availability")

    hourly = (
        df.select(
            pl.col("hours_sale")
            .list.slice(6, 16)
            .alias("hourly_sales"),

            pl.col("hours_stock_status")
            .list.slice(6, 16)
            .alias("stock_status"),
        )
        .explode(["hourly_sales", "stock_status"])
    )

    hourly_analysis = (
        hourly.group_by("stock_status")
        .agg(
            pl.len().alias("hourly_observations"),
            pl.col("hourly_sales").mean().alias("mean_hourly_sales"),
            pl.col("hourly_sales").median().alias("median_hourly_sales"),
            pl.col("hourly_sales").sum().alias("total_hourly_sales"),
            (pl.col("hourly_sales") == 0)
            .mean()
            .alias("zero_sales_percentage"),
        )
        .sort("stock_status")
        .collect()
    )

    print(hourly_analysis)

    # 3. Hourly stockout frequency
    print("\n3. Stockout frequency by hour")

    hourly_frequency = (
        df.select(
            pl.col("hours_stock_status")
            .list.slice(6, 16)
            .alias("stock_status")
        )
        .with_columns(
            pl.int_ranges(6, 22).alias("hour")
        )
        .explode(["hour", "stock_status"])
        .group_by("hour")
        .agg(
            pl.len().alias("observations"),
            (pl.col("stock_status") == 1)
            .mean()
            .alias("stockout_percentage"),
        )
        .sort("hour")
        .collect()
    )

    print(hourly_frequency)


def main():
    for name, path in FILES.items():
        analyze_stockouts(name, path)


if __name__ == "__main__":
    main()
