
from pathlib import Path

import polars as pl

DATA_DIR = Path("data/raw/freshretailnet")

FILES = {
    "Train": DATA_DIR / "data/train.parquet",
    "Evaluation": DATA_DIR / "data/eval.parquet",
}


def profile_dataset(name: str, path: Path):
    print(f"\n{'=' * 60}")
    print(f"{name} DATA PROFILE")
    print(f"{'=' * 60}")

    df = pl.scan_parquet(path)

    # 1. Date coverage
    print("\n1. Date coverage")

    print(
        df.select(
            pl.col("dt").min().alias("start_date"),
            pl.col("dt").max().alias("end_date"),
            pl.col("dt").n_unique().alias("unique_dates"),
        ).collect()
    )

    # 2. Unique entities
    print("\n2. Unique entities")

    print(
        df.select(
            pl.col("store_id").n_unique().alias("stores"),
            pl.col("product_id").n_unique().alias("products"),
            pl.struct(["store_id", "product_id"])
            .n_unique()
            .alias("store_product_pairs"),
            pl.col("city_id").n_unique().alias("cities"),
        ).collect()
    )

    # 3. Missing values
    print("\n3. Missing values")

    missing = df.select(
        [
            pl.col(column).null_count().alias(column)
            for column in df.collect_schema().names()
        ]
    ).collect()

    print(missing)

    # 4. Sales distribution
    print("\n4. Sales distribution")

    print(
        df.select(
            pl.col("sale_amount").min().alias("minimum"),
            pl.col("sale_amount").max().alias("maximum"),
            pl.col("sale_amount").mean().alias("mean"),
            pl.col("sale_amount").median().alias("median"),
            pl.col("sale_amount").std().alias("std_dev"),
        ).collect()
    )

    # 5. Stock availability
    print("\n5. Stock availability")

    print(
        df.select(
            pl.col("stock_hour6_22_cnt").min().alias("minimum"),
            pl.col("stock_hour6_22_cnt").max().alias("maximum"),
            pl.col("stock_hour6_22_cnt").mean().alias("mean"),
            pl.col("stock_hour6_22_cnt").median().alias("median"),
        ).collect()
    )

    print("\nStock status values:")

    print(
        df.select(
            pl.col("hours_stock_status").explode().unique().sort()
        ).collect()
    )

    # 6. Hourly array lengths
    print("\n6. Hourly array lengths")

    print(
        df.select(
            pl.col("hours_sale").list.len().min().alias("min_sales_length"),
            pl.col("hours_sale").list.len().max().alias("max_sales_length"),
            pl.col("hours_stock_status")
            .list.len()
            .min()
            .alias("min_stock_length"),
            pl.col("hours_stock_status")
            .list.len()
            .max()
            .alias("max_stock_length"),
        ).collect()
    )

    # 7. Duplicate records
    print("\n7. Duplicate store-product-date records")

    duplicates = (
        df.group_by(["store_id", "product_id", "dt"])
        .len()
        .filter(pl.col("len") > 1)
        .select(pl.len())
        .collect()
        .item()
    )

    print(f"Duplicate groups: {duplicates:,}")


def main():
    for name, path in FILES.items():
        profile_dataset(name, path)


if __name__ == "__main__":
    main()
