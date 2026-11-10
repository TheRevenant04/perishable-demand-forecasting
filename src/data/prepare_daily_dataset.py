
from datetime import timedelta
from pathlib import Path

import polars as pl

DATA_DIR = Path("data/raw/freshretailnet/data")
OUTPUT_DIR = Path("data/processed")
OUTPUT_PATH = OUTPUT_DIR / "daily_forecasting.parquet"

TRAIN_PATH = DATA_DIR / "train.parquet"
EVAL_PATH = DATA_DIR / "eval.parquet"

GROUP_KEYS = ["store_id", "product_id"]

FORECAST_HORIZON = 7

SALES_LAGS = [1, 2, 3, 4, 5, 6, 7, 14]


def load_dataset(path: Path, split: str) -> pl.LazyFrame:
    """Load the daily records."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return (
        pl.scan_parquet(path)
        .select(
            "city_id",
            "store_id",
            "management_group_id",
            "first_category_id",
            "second_category_id",
            "third_category_id",
            "product_id",
            "dt",
            "sale_amount",
            "stock_hour6_22_cnt",
            "holiday_flag",
        )
        .with_columns(
            pl.col("dt")
            .str.strptime(
                pl.Date,
                format="%Y-%m-%d",
                strict=True,
            )
            .alias("origin_date"),

            pl.lit(split).alias("source_split"),
        )
    )


def get_date_boundaries() -> tuple:
    """Identify the chronological boundaries of the datasets."""

    train_dates = (
        pl.scan_parquet(TRAIN_PATH)
        .select(
            pl.col("dt")
            .str.strptime(pl.Date, format="%Y-%m-%d")
            .min()
            .alias("first_date"),

            pl.col("dt")
            .str.strptime(pl.Date, format="%Y-%m-%d")
            .max()
            .alias("last_date"),
        )
        .collect()
        .row(0)
    )

    eval_dates = (
        pl.scan_parquet(EVAL_PATH)
        .select(
            pl.col("dt")
            .str.strptime(pl.Date, format="%Y-%m-%d")
            .min()
            .alias("first_date"),

            pl.col("dt")
            .str.strptime(pl.Date, format="%Y-%m-%d")
            .max()
            .alias("last_date"),
        )
        .collect()
        .row(0)
    )

    return (
        train_dates[0],
        train_dates[1],
        eval_dates[0],
        eval_dates[1],
    )


def create_historical_features(
    data: pl.LazyFrame,
) -> pl.LazyFrame:
    """Create features using information available at the forecast origin."""

    data = data.sort(GROUP_KEYS + ["origin_date"])

    expressions = []

    for lag in SALES_LAGS:
        expressions.append(
            pl.col("sale_amount")
            .shift(lag)
            .over(GROUP_KEYS)
            .alias(f"sales_lag_{lag}")
        )

    expressions.extend(
        [
            pl.col("stock_hour6_22_cnt")
            .shift(1)
            .over(GROUP_KEYS)
            .alias("stockout_hours_lag_1"),

            pl.col("stock_hour6_22_cnt")
            .shift(7)
            .over(GROUP_KEYS)
            .alias("stockout_hours_lag_7"),
        ]
    )

    data = data.with_columns(expressions)

    # Calculate averages from previous observations only.
    data = data.with_columns(
        pl.mean_horizontal(
            [
                pl.col(f"sales_lag_{lag}")
                for lag in range(1, 8)
            ]
        ).alias("sales_rolling_mean_7"),

        pl.mean_horizontal(
            [
                pl.col(f"sales_lag_{lag}")
                for lag in range(1, 15)
            ]
        ).alias("sales_rolling_mean_14"),
    )

    return data


def create_horizon_dataset(
    data: pl.LazyFrame,
    horizon: int,
) -> pl.LazyFrame:
    """Create prediction rows for one forecast horizon."""

    horizon_data = data.with_columns(
        pl.col("sale_amount")
        .shift(-horizon)
        .over(GROUP_KEYS)
        .alias("target_sales"),

        pl.col("stock_hour6_22_cnt")
        .shift(-horizon)
        .over(GROUP_KEYS)
        .alias("target_stockout_hours"),

        pl.col("origin_date")
        .shift(-horizon)
        .over(GROUP_KEYS)
        .alias("target_date"),

        pl.col("holiday_flag")
        .shift(-horizon)
        .over(GROUP_KEYS)
        .alias("target_holiday_flag"),
    )

    horizon_data = horizon_data.with_columns(
        pl.lit(horizon)
        .cast(pl.Int8)
        .alias("forecast_horizon")
    )

    horizon_data = horizon_data.with_columns(
        pl.col("target_date")
        .dt.weekday()
        .cast(pl.Int8)
        .alias("target_weekday"),

        pl.col("target_date")
        .dt.month()
        .cast(pl.Int8)
        .alias("target_month"),

        (
            pl.col("target_stockout_hours") == 0
        ).alias("target_is_uncensored"),
    )

    return horizon_data.select(
        "city_id",
        "store_id",
        "management_group_id",
        "first_category_id",
        "second_category_id",
        "third_category_id",
        "product_id",
        "origin_date",
        "forecast_horizon",
        "target_date",

        *[
            f"sales_lag_{lag}"
            for lag in SALES_LAGS
        ],

        "sales_rolling_mean_7",
        "sales_rolling_mean_14",
        "stockout_hours_lag_1",
        "stockout_hours_lag_7",

        "target_weekday",
        "target_month",
        "target_holiday_flag",

        "target_sales",
        "target_stockout_hours",
        "target_is_uncensored",
    )


def create_forecasting_dataset(
    data: pl.LazyFrame,
) -> pl.LazyFrame:
    """Create a dataset containing all seven forecast horizons."""

    horizon_datasets = [
        create_horizon_dataset(data, horizon)
        for horizon in range(1, FORECAST_HORIZON + 1)
    ]

    return pl.concat(horizon_datasets)


def assign_splits(
    data: pl.LazyFrame,
    train_end,
) -> pl.LazyFrame:
    """
    Assign chronological training, validation and test partitions.

    Validation:
        Seven-day forecast originating seven days before train_end.

    Test:
        Seven-day forecast originating on train_end.

    Training:
        Historical target dates ending before validation starts.
    """

    validation_origin = train_end - timedelta(
        days=FORECAST_HORIZON
    )

    return data.with_columns(
        pl.when(
            pl.col("target_date") <= validation_origin
        )
        .then(pl.lit("train"))

        .when(
            pl.col("origin_date") == validation_origin
        )
        .then(pl.lit("validation"))

        .when(
            pl.col("origin_date") == train_end
        )
        .then(pl.lit("test"))

        .otherwise(None)
        .alias("dataset_split")
    ).filter(
        pl.col("dataset_split").is_not_null()
    )


def print_dataset_summary(path: Path) -> None:
    """Print dataset statistics without loading everything into memory."""

    data = pl.scan_parquet(path)

    summary = data.select(
        pl.len().alias("total_rows"),

        (pl.col("dataset_split") == "train")
        .sum()
        .alias("training_rows"),

        (pl.col("dataset_split") == "validation")
        .sum()
        .alias("validation_rows"),

        (pl.col("dataset_split") == "test")
        .sum()
        .alias("test_rows"),

        pl.col("target_is_uncensored")
        .sum()
        .alias("uncensored_targets"),

        (~pl.col("target_is_uncensored"))
        .sum()
        .alias("stockout_affected_targets"),

        pl.col("forecast_horizon")
        .min()
        .alias("minimum_horizon"),

        pl.col("forecast_horizon")
        .max()
        .alias("maximum_horizon"),

        pl.col("origin_date")
        .min()
        .alias("first_origin_date"),

        pl.col("origin_date")
        .max()
        .alias("last_origin_date"),
    ).collect()

    print("\nDataset summary:")
    print(summary)

    print("\nRows by split and horizon:")

    horizon_summary = (
        data.group_by(
            "dataset_split",
            "forecast_horizon",
        )
        .agg(
            pl.len().alias("rows"),

            pl.col("target_is_uncensored")
            .sum()
            .alias("uncensored_targets"),
        )
        .sort(
            ["dataset_split", "forecast_horizon"]
        )
        .collect()
    )

    print(horizon_summary)

    print("\nDataset schema:")
    print(data.collect_schema())


def prepare_dataset() -> None:
    """Build the seven-day forecasting dataset."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        train_start,
        train_end,
        eval_start,
        eval_end,
    ) = get_date_boundaries()

    print("Dataset date boundaries:")
    print(f"Training:   {train_start} to {train_end}")
    print(f"Evaluation: {eval_start} to {eval_end}")

    validation_origin = train_end - timedelta(
        days=FORECAST_HORIZON
    )

    print(f"Validation forecast origin: {validation_origin}")
    print(f"Test forecast origin:       {train_end}")

    if eval_start != train_end + timedelta(days=1):
        raise ValueError(
            "Training and evaluation dates are not contiguous."
        )

    if eval_end != train_end + timedelta(days=FORECAST_HORIZON):
        raise ValueError(
            "Evaluation period does not match the seven-day horizon."
        )

    print("\nLoading datasets...")

    train = load_dataset(
        TRAIN_PATH,
        "train",
    )

    evaluation = load_dataset(
        EVAL_PATH,
        "evaluation",
    )

    combined = pl.concat(
        [train, evaluation]
    )

    print("Creating historical features...")

    featured = create_historical_features(
        combined
    )

    print("Creating seven forecast horizons...")

    forecasting_data = create_forecasting_dataset(
        featured
    )

    print("Assigning chronological splits...")

    forecasting_data = assign_splits(
        forecasting_data,
        train_end,
    )

    # Keep only rows with sufficient historical information
    # and a known target.
    forecasting_data = forecasting_data.filter(
        pl.col("sales_lag_14").is_not_null()
        & pl.col("target_sales").is_not_null()
        & pl.col("target_stockout_hours").is_not_null()
        & pl.col("target_date").is_not_null()
    )

    print(f"\nSaving dataset to: {OUTPUT_PATH}")

    forecasting_data.sink_parquet(
        OUTPUT_PATH,
        compression="zstd",
    )

    print("\nDataset preparation completed.")

    print_dataset_summary(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    prepare_dataset()
