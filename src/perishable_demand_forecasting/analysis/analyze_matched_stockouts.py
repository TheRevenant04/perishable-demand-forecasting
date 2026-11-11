
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATA_DIR = Path("data/raw/freshretailnet/data")

FILES = {
    "Train": DATA_DIR / "train.parquet",
    "Evaluation": DATA_DIR / "eval.parquet",
}

# Require a minimum number of daily observations in each group.
MIN_OBSERVATIONS = {
    "Train": 5,
    "Evaluation": 2,
}


# ---------------------------------------------------------------------
# Daily matched stockout analysis
# ---------------------------------------------------------------------

def analyse_matched_store_products(
    name: str,
    path: Path,
) -> None:
    """
    Compare daily observed sales on fully in-stock and
    stockout-affected days for the same store-product combinations.
    """

    print("\n" + "=" * 75)
    print(f"{name.upper()} DAILY MATCHED STOCKOUT ANALYSIS")
    print("=" * 75)

    if not path.exists():
        print(f"Dataset not found: {path}")
        return

    min_observations = MIN_OBSERVATIONS[name]

    print(
        f"\nMinimum daily observations per stock condition: "
        f"{min_observations}"
    )

    df = pl.scan_parquet(path).select(
        [
            "store_id",
            "product_id",
            "dt",
            "sale_amount",
            "stock_hour6_22_cnt",
        ]
    )

    # -------------------------------------------------------------
    # 1. Aggregate daily observations for each store-product pair
    # -------------------------------------------------------------

    pair_summary = (
        df.group_by(
            [
                "store_id",
                "product_id",
            ]
        )
        .agg(
            (
                pl.col("stock_hour6_22_cnt") == 0
            )
            .sum()
            .alias("in_stock_days"),

            (
                pl.col("stock_hour6_22_cnt") > 0
            )
            .sum()
            .alias("stockout_days"),

            pl.col("sale_amount")
            .filter(
                pl.col("stock_hour6_22_cnt") == 0
            )
            .mean()
            .alias("mean_sales_in_stock"),

            pl.col("sale_amount")
            .filter(
                pl.col("stock_hour6_22_cnt") > 0
            )
            .mean()
            .alias("mean_sales_stockout"),

            pl.col("sale_amount")
            .filter(
                pl.col("stock_hour6_22_cnt") == 0
            )
            .sum()
            .alias("total_sales_in_stock"),

            pl.col("sale_amount")
            .filter(
                pl.col("stock_hour6_22_cnt") > 0
            )
            .sum()
            .alias("total_sales_stockout"),
        )
    )

    # -------------------------------------------------------------
    # 2. Keep store-product pairs with enough observations
    # -------------------------------------------------------------

    matched = (
        pair_summary
        .filter(
            (pl.col("in_stock_days") >= min_observations)
            & (pl.col("stockout_days") >= min_observations)
        )
        .with_columns(
            (
                pl.col("mean_sales_in_stock")
                - pl.col("mean_sales_stockout")
            ).alias("mean_sales_difference"),

            (
                pl.col("mean_sales_stockout")
                < pl.col("mean_sales_in_stock")
            ).alias("stockout_sales_lower"),
        )
        .collect()
    )

    print(
        f"\nMatched store-product combinations: "
        f"{matched.height:,}"
    )

    if matched.is_empty():
        print(
            "No store-product combinations met the minimum "
            "observation requirements."
        )
        return

    # -------------------------------------------------------------
    # 3. Summarise matched daily sales differences
    # -------------------------------------------------------------

    print("\nWithin-store-product daily comparison:")

    comparison = matched.select(
        pl.len().alias("matched_groups"),

        pl.col("mean_sales_in_stock")
        .mean()
        .alias("avg_daily_sales_in_stock"),

        pl.col("mean_sales_stockout")
        .mean()
        .alias("avg_daily_sales_stockout"),

        pl.col("mean_sales_difference")
        .mean()
        .alias("avg_daily_sales_difference"),

        pl.col("mean_sales_difference")
        .median()
        .alias("median_daily_sales_difference"),

        pl.col("mean_sales_difference")
        .std()
        .alias("std_daily_sales_difference"),

        pl.col("stockout_sales_lower")
        .mean()
        .alias("percentage_groups_lower_during_stockouts"),
    )

    print(comparison)

    # -------------------------------------------------------------
    # 4. Distribution of matched differences
    # -------------------------------------------------------------

    print("\nDistribution of store-product daily sales differences:")

    distribution = matched.select(
        pl.col("mean_sales_difference")
        .min()
        .alias("minimum_difference"),

        pl.col("mean_sales_difference")
        .quantile(0.25)
        .alias("q25_difference"),

        pl.col("mean_sales_difference")
        .median()
        .alias("median_difference"),

        pl.col("mean_sales_difference")
        .quantile(0.75)
        .alias("q75_difference"),

        pl.col("mean_sales_difference")
        .max()
        .alias("maximum_difference"),
    )

    print(distribution)

    # -------------------------------------------------------------
    # 5. Compare observation counts
    # -------------------------------------------------------------

    print("\nMatched observation counts:")

    observation_summary = matched.select(
        pl.col("in_stock_days")
        .sum()
        .alias("total_in_stock_days"),

        pl.col("stockout_days")
        .sum()
        .alias("total_stockout_affected_days"),

        pl.col("in_stock_days")
        .mean()
        .alias("avg_in_stock_days_per_group"),

        pl.col("stockout_days")
        .mean()
        .alias("avg_stockout_days_per_group"),
    )

    print(observation_summary)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    for name, path in FILES.items():
        analyse_matched_store_products(name, path)


if __name__ == "__main__":
    main()
