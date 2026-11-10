
from pathlib import Path

import polars as pl

DATA_DIR = Path("data/raw/freshretailnet")

FILES = {
    "Train": DATA_DIR / "data/train.parquet",
    "Evaluation": DATA_DIR / "data/eval.parquet",
}

MIN_OBSERVATIONS = {
    "Train": 5,
    "Evaluation": 2,
}


def analyze_matched_stockouts(name: str, path: Path):
    print(f"\n{'=' * 70}")
    print(f"{name.upper()} MATCHED STOCKOUT ANALYSIS")
    print(f"{'=' * 70}")

    min_observations = MIN_OBSERVATIONS[name]

    print(
        f"\nMinimum observations per stock status: "
        f"{min_observations}"
    )

    df = pl.scan_parquet(path)

    hourly = (
        df.select(
            "store_id",
            "product_id",
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
        .explode(
            ["hour", "sales", "stock_status"]
        )
    )

    matched = (
        hourly.group_by(
            ["store_id", "product_id", "hour"]
        )
        .agg(
            pl.col("sales")
            .filter(pl.col("stock_status") == 0)
            .mean()
            .alias("mean_in_stock"),

            pl.col("sales")
            .filter(pl.col("stock_status") == 1)
            .mean()
            .alias("mean_out_of_stock"),

            (pl.col("stock_status") == 0)
            .sum()
            .alias("in_stock_observations"),

            (pl.col("stock_status") == 1)
            .sum()
            .alias("out_of_stock_observations"),
        )
        .filter(
            (
                pl.col("in_stock_observations")
                >= min_observations
            )
            & (
                pl.col("out_of_stock_observations")
                >= min_observations
            )
        )
        .with_columns(
            (
                pl.col("mean_out_of_stock")
                - pl.col("mean_in_stock")
            ).alias("sales_difference")
        )
        .collect()
    )

    print(
        f"\nMatched store-product-hour groups: "
        f"{matched.height:,}"
    )

    if matched.is_empty():
        print(
            "\nNo matched groups found. "
            "Consider reviewing the observation threshold."
        )
        return

    summary = (
        matched.group_by("hour")
        .agg(
            pl.len().alias("matched_groups"),

            pl.col("mean_in_stock")
            .mean()
            .alias("avg_in_stock_sales"),

            pl.col("mean_out_of_stock")
            .mean()
            .alias("avg_out_of_stock_sales"),

            pl.col("sales_difference")
            .mean()
            .alias("avg_sales_difference"),

            pl.col("sales_difference")
            .median()
            .alias("median_sales_difference"),
        )
        .sort("hour")
    )

    print("\nWithin-product hourly comparison:")
    print(summary)


def main():
    for name, path in FILES.items():
        analyze_matched_stockouts(name, path)


if __name__ == "__main__":
    main()
