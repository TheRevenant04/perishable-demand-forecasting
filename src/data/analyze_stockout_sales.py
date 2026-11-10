
from pathlib import Path

import polars as pl

DATA_DIR = Path("data/raw/freshretailnet")

FILES = {
    "Train": DATA_DIR / "data/train.parquet",
    "Evaluation": DATA_DIR / "data/eval.parquet",
}


def analyze_stockout_sales(name: str, path: Path):
    print(f"\n{'=' * 70}")
    print(f"{name.upper()} NONZERO SALES DURING STOCKOUTS")
    print(f"{'=' * 70}")

    df = pl.scan_parquet(path)

    hourly = (
        df.select(
            "store_id",
            "product_id",
            "dt",
            pl.col("hours_sale").list.slice(6, 16).alias("sales"),
            pl.col("hours_stock_status")
            .list.slice(6, 16)
            .alias("stock_status"),
        )
        .with_columns(
            pl.int_ranges(6, 22).alias("hour")
        )
        .explode(["hour", "sales", "stock_status"])
    )

    stockout_sales = (
        hourly
        .filter(
            (pl.col("stock_status") == 1)
            & (pl.col("sales") > 0)
        )
    )

    summary = stockout_sales.select(
        pl.len().alias("nonzero_stockout_records"),
        pl.col("sales").sum().alias("total_recorded_sales"),
        pl.col("sales").mean().alias("mean_sales"),
        pl.col("sales").median().alias("median_sales"),
        pl.col("sales").max().alias("max_sales"),
    ).collect()

    total_stockout_records = (
        hourly
        .filter(pl.col("stock_status") == 1)
        .select(pl.len())
        .collect()
        .item()
    )

    print("\nOverall summary:")
    print(summary)

    nonzero_count = summary["nonzero_stockout_records"].item()

    percentage = (
        nonzero_count / total_stockout_records * 100
        if total_stockout_records
        else 0
    )

    print(f"\nTotal stockout records: {total_stockout_records:,}")
    print(f"Nonzero stockout records: {nonzero_count:,}")
    print(f"Percentage with nonzero sales: {percentage:.4f}%")

    hourly_summary = (
        stockout_sales
        .group_by("hour")
        .agg(
            pl.len().alias("nonzero_records"),
            pl.col("sales").mean().alias("mean_sales"),
            pl.col("sales").median().alias("median_sales"),
            pl.col("sales").max().alias("max_sales"),
        )
        .sort("hour")
        .collect()
    )

    print("\nNonzero stockout sales by hour:")
    print(hourly_summary)

    samples = (
        stockout_sales
        .select(
            "store_id",
            "product_id",
            "dt",
            "hour",
            "sales",
            "stock_status",
        )
        .sort("sales", descending=True)
        .limit(20)
        .collect()
    )

    print("\n20 examples with the highest recorded sales:")
    print(samples)


def main():
    for name, path in FILES.items():
        analyze_stockout_sales(name, path)


if __name__ == "__main__":
    main()
