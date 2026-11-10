
from pathlib import Path

import polars as pl

DATA_DIR = Path("data/raw/freshretailnet")

FILES = {
    "Train": DATA_DIR / "data/train.parquet",
    "Evaluation": DATA_DIR / "data/eval.parquet",
}


def validate_stockouts(name: str, path: Path):
    print(f"\n{'=' * 60}")
    print(f"{name} STOCKOUT VALIDATION")
    print(f"{'=' * 60}")

    df = pl.scan_parquet(path)

    # Compare daily sales with the sum of hourly sales.
    # Compare daytime stockout count with hourly stock status.
    checks = df.select(
        (
            pl.col("hours_sale").list.sum()
            - pl.col("sale_amount")
        ).abs().max().alias("max_sales_difference"),

        (
            pl.col("hours_stock_status")
            .list.slice(6, 16)
            .list.sum()
            - pl.col("stock_hour6_22_cnt")
        ).abs().max().alias("max_stockout_difference"),

        (pl.col("stock_hour6_22_cnt") > 0)
        .sum()
        .alias("records_with_stockouts"),

        (pl.col("stock_hour6_22_cnt") == 0)
        .sum()
        .alias("records_without_stockouts"),

        (pl.col("stock_hour6_22_cnt") == 16)
        .sum()
        .alias("fully_stocked_out_records"),
    ).collect()

    print("\nValidation results:")
    print(checks)

    # Distribution of stockout hours.
    print("\nStockout hour distribution:")

    distribution = (
        df.group_by("stock_hour6_22_cnt")
        .len()
        .sort("stock_hour6_22_cnt")
        .collect()
    )

    print(distribution)

    # Stockout status distribution across all 24 hourly positions.
    print("\nHourly stock status distribution:")

    status_distribution = (
        df.select(
            pl.col("hours_stock_status").explode()
        )
        .group_by("hours_stock_status")
        .len()
        .sort("hours_stock_status")
        .collect()
    )

    print(status_distribution)


def main():
    for name, path in FILES.items():
        validate_stockouts(name, path)


if __name__ == "__main__":
    main()
