
from pathlib import Path

import polars as pl

DATA_DIR = Path("data/raw/freshretailnet")

FILES = {
    "Train": DATA_DIR / "data/train.parquet",
    "Evaluation": DATA_DIR / "data/eval.parquet",
}


def analyze_hourly_demand(name: str, path: Path):
    print(f"\n{'=' * 70}")
    print(f"{name} HOURLY DEMAND PROFILE")
    print(f"{'=' * 70}")

    df = pl.scan_parquet(path)

    hourly = (
        df.select(
            pl.col("hours_sale")
            .list.slice(6, 16)
            .alias("sales"),

            pl.col("hours_stock_status")
            .list.slice(6, 16)
            .alias("stock_status"),
        )
        .with_columns(
            pl.int_ranges(6, 22).alias("hour")
        )
        .explode(["hour", "sales", "stock_status"])
    )

    # Demand and availability by hour and stock status
    profile = (
        hourly.group_by(["hour", "stock_status"])
        .agg(
            pl.len().alias("observations"),
            pl.col("sales").mean().alias("mean_sales"),
            pl.col("sales").median().alias("median_sales"),
            (pl.col("sales") == 0)
            .mean()
            .alias("zero_sales_percentage"),
        )
        .sort(["hour", "stock_status"])
        .collect()
    )

    print("\nHourly sales by stock status:")
    print(profile)

    # Overall hourly profile
    overall = (
        hourly.group_by("hour")
        .agg(
            pl.col("sales").mean().alias("mean_sales"),
            (pl.col("stock_status") == 1)
            .mean()
            .alias("stockout_percentage"),
        )
        .sort("hour")
        .collect()
    )

    print("\nOverall hourly profile:")
    print(overall)


def main():
    for name, path in FILES.items():
        analyze_hourly_demand(name, path)


if __name__ == "__main__":
    main()
