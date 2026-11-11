
import os

# Must be configured before importing Polars.
os.environ.setdefault("POLARS_MAX_THREADS", "8")

from datetime import timedelta
from math import floor
from pathlib import Path

import joblib
import numpy as np
import polars as pl

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATA_DIR = Path("data/raw/freshretailnet/data")
OUTPUT_DIR = Path("data/processed/freshretailnet/forecasting")

TRAIN_FILE = DATA_DIR / "train.parquet"
EVAL_FILE = DATA_DIR / "eval.parquet"

MODEL_DIR = Path("models/recovery/regularization")

RIDGE_ALPHA_1 = MODEL_DIR / "G_business_hourly_alpha_1.joblib"
RIDGE_ALPHA_10 = MODEL_DIR / "G_business_hourly_alpha_10.joblib"

FORECAST_HORIZON = 7
EMBARGO_DAYS = FORECAST_HORIZON

TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20
TEST_RATIO = 0.20

ID_COLUMNS = [
    "city_id",
    "store_id",
    "management_group_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
    "product_id",
]

GROUP_COLUMNS = ["store_id", "product_id"]

RECOVERY_FEATURES = [
    "simulated_observed_daily_sales",
    "simulated_stockout_hours",
    "city_id",
    "store_id",
    "product_id",
    "day_of_week",
    "month",
    "discount",
    "holiday_flag",
    "activity_flag",
] + [
    f"observed_hour_{hour:02d}"
    for hour in range(6, 22)
]

CATEGORICAL_RECOVERY_FEATURES = [
    "city_id",
    "store_id",
    "product_id",
]

DEMAND_LAGS = list(range(1, 15))
STOCKOUT_LAGS = [1, 7]

FORECAST_FEATURES = (
    [f"demand_lag_{lag}" for lag in DEMAND_LAGS]
    + [
        "demand_rolling_mean_7",
        "demand_rolling_mean_14",
    ]
    + [
        f"stockout_hours_lag_{lag}"
        for lag in STOCKOUT_LAGS
    ]
)


# ---------------------------------------------------------------------
# Date handling
# ---------------------------------------------------------------------

def parse_date_column(
    df: pl.LazyFrame,
    date_column: str = "dt",
) -> pl.LazyFrame:
    schema = df.collect_schema()
    dtype = schema[date_column]

    if dtype == pl.String:
        return df.with_columns(
            pl.col(date_column)
            .str.strptime(
                pl.Date,
                "%Y-%m-%d",
                strict=True,
            )
            .alias(date_column)
        )

    return df.with_columns(
        pl.col(date_column).cast(pl.Date)
    )


# ---------------------------------------------------------------------
# Load source datasets
# ---------------------------------------------------------------------

def load_source_data() -> pl.LazyFrame:
    print("Loading source datasets...")

    required_columns = (
        ID_COLUMNS
        + [
            "dt",
            "sale_amount",
            "stock_hour6_22_cnt",
            "holiday_flag",
            "discount",
            "activity_flag",
            "hours_sale",
        ]
    )

    train = parse_date_column(
        pl.scan_parquet(TRAIN_FILE).select(required_columns)
    )

    evaluation = parse_date_column(
        pl.scan_parquet(EVAL_FILE).select(required_columns)
    )

    return (
        pl.concat(
            [train, evaluation],
            how="vertical",
        )
        .sort(GROUP_COLUMNS + ["dt"])
    )


# ---------------------------------------------------------------------
# Hourly feature extraction
# ---------------------------------------------------------------------

def extract_hourly_features(
    df: pl.LazyFrame,
) -> pl.LazyFrame:
    print("Extracting hourly observations...")

    hours_dtype = df.collect_schema()["hours_sale"]

    if isinstance(hours_dtype, pl.List):
        get_hour = lambda hour: pl.col("hours_sale").list.get(hour)

    elif isinstance(hours_dtype, pl.Array):
        get_hour = lambda hour: pl.col("hours_sale").arr.get(hour)

    else:
        raise TypeError(
            f"Unsupported hours_sale type: {hours_dtype}"
        )

    hourly_expressions = [
        get_hour(hour).alias(f"observed_hour_{hour:02d}")
        for hour in range(6, 22)
    ]

    return df.with_columns(hourly_expressions)


# ---------------------------------------------------------------------
# Recovery feature preparation
# ---------------------------------------------------------------------

def prepare_recovery_features(
    df: pl.DataFrame,
) -> pl.DataFrame:
    result = df.select(
        pl.col("sale_amount").alias(
            "simulated_observed_daily_sales"
        ),
        pl.col("stock_hour6_22_cnt").alias(
            "simulated_stockout_hours"
        ),
        "city_id",
        "store_id",
        "product_id",
        pl.col("dt").dt.weekday().alias("day_of_week"),
        pl.col("dt").dt.month().alias("month"),
        "discount",
        "holiday_flag",
        "activity_flag",
        *[
            f"observed_hour_{hour:02d}"
            for hour in range(6, 22)
        ],
    )

    return result.with_columns(
        [
            pl.col(column).cast(pl.String)
            for column in CATEGORICAL_RECOVERY_FEATURES
        ]
    )


# ---------------------------------------------------------------------
# Demand recovery
# ---------------------------------------------------------------------

def load_recovery_models():
    print("Loading frozen recovery models...")

    for path in [RIDGE_ALPHA_1, RIDGE_ALPHA_10]:
        if not path.exists():
            raise FileNotFoundError(
                f"Recovery model not found: {path}"
            )

    model_alpha_1 = joblib.load(RIDGE_ALPHA_1)
    model_alpha_10 = joblib.load(RIDGE_ALPHA_10)

    return model_alpha_1, model_alpha_10


def recover_daily_demand(
    df: pl.DataFrame,
    model_alpha_1,
    model_alpha_10,
) -> pl.DataFrame:
    """
    Apply the frozen hybrid recovery strategy.

    0-5 stockout hours: observed daily sales
    6-9 stockout hours: Ridge alpha 10
    10-16 stockout hours: Ridge alpha 1

    Apply non-negative clipping after hybrid recovery.
    """

    recovery_features = prepare_recovery_features(df)

    missing = [
        column
        for column in RECOVERY_FEATURES
        if column not in recovery_features.columns
    ]

    if missing:
        raise ValueError(
            f"Missing recovery features: {missing}"
        )

    X = recovery_features.select(
        RECOVERY_FEATURES
    ).to_pandas()

    alpha_1_predictions = model_alpha_1.predict(X)
    alpha_10_predictions = model_alpha_10.predict(X)

    observed_sales = df["sale_amount"].to_numpy()
    stockout_hours = df["stock_hour6_22_cnt"].to_numpy()

    recovered = np.select(
        [
            stockout_hours <= 5,
            (stockout_hours >= 6) & (stockout_hours <= 9),
            (stockout_hours >= 10) & (stockout_hours <= 16),
        ],
        [
            observed_sales,
            alpha_10_predictions,
            alpha_1_predictions,
        ],
        default=np.nan,
    )

    if not np.isfinite(recovered).all():
        raise ValueError(
            "Demand recovery generated non-finite values."
        )

    negative_count = int((recovered < 0).sum())

    # Demand cannot be negative.
    recovered = np.maximum(recovered, 0.0)

    if not np.isfinite(recovered).all():
        raise ValueError(
            "Recovered demand contains non-finite values."
        )

    if (recovered < 0).any():
        raise ValueError(
            "Negative recovered demand remains after clipping."
        )

    if negative_count:
        print(
            f"Clipped {negative_count:,} negative demand "
            f"predictions to zero."
        )

    return df.select(
        ID_COLUMNS + [
            "dt",
            "stock_hour6_22_cnt",
            "holiday_flag",
        ]
    ).with_columns(
        pl.Series(
            "recovered_daily_demand",
            recovered,
        )
    )


def build_recovered_daily_dataset(
    source_df: pl.LazyFrame,
    model_alpha_1,
    model_alpha_10,
) -> pl.DataFrame:
    print("\n" + "=" * 65)
    print("RECOVERING HISTORICAL DAILY DEMAND")
    print("=" * 65)

    featured = extract_hourly_features(source_df)

    data = featured.collect()

    print(f"Source observations: {data.height:,}")

    daily_partitions = data.partition_by(
        "dt",
        maintain_order=True,
    )

    recovered_parts = []

    total_dates = len(daily_partitions)

    for index, daily_data in enumerate(
        daily_partitions,
        start=1,
    ):
        recovered_daily = recover_daily_demand(
            daily_data,
            model_alpha_1,
            model_alpha_10,
        )

        recovered_parts.append(recovered_daily)

        if index % 10 == 0 or index == total_dates:
            print(
                f"Processed {index:,}/{total_dates:,} dates"
            )

    recovered = (
        pl.concat(recovered_parts)
        .sort(GROUP_COLUMNS + ["dt"])
    )

    print(f"Recovered observations: {recovered.height:,}")

    # Final dataset-level validation.
    summary = recovered.select(
        pl.col("recovered_daily_demand").min().alias("minimum"),
        pl.col("recovered_daily_demand").max().alias("maximum"),
        pl.col("recovered_daily_demand").null_count().alias("null_count"),
    )

    stats = summary.row(0, named=True)

    if stats["minimum"] < 0:
        raise ValueError(
            "Recovered dataset contains negative demand."
        )

    if stats["null_count"] > 0:
        raise ValueError(
            "Recovered dataset contains null demand values."
        )

    print(
        "Recovered demand minimum: "
        f"{stats['minimum']:.6f}"
    )
    print(
        "Recovered demand maximum: "
        f"{stats['maximum']:.6f}"
    )

    return recovered


# ---------------------------------------------------------------------
# Historical forecasting features
# ---------------------------------------------------------------------

def create_features(
    df: pl.LazyFrame,
) -> pl.LazyFrame:
    print("\nCreating historical forecasting features...")

    demand_lags = [
        pl.col("recovered_daily_demand")
        .shift(lag)
        .over(GROUP_COLUMNS)
        .alias(f"demand_lag_{lag}")
        for lag in DEMAND_LAGS
    ]

    stockout_lags = [
        pl.col("stock_hour6_22_cnt")
        .shift(lag)
        .over(GROUP_COLUMNS)
        .alias(f"stockout_hours_lag_{lag}")
        for lag in STOCKOUT_LAGS
    ]

    df = df.with_columns(
        demand_lags + stockout_lags
    )

    lag_7 = [
        pl.col(f"demand_lag_{lag}")
        for lag in range(1, 8)
    ]

    lag_14 = [
        pl.col(f"demand_lag_{lag}")
        for lag in range(1, 15)
    ]

    return df.with_columns(
        pl.mean_horizontal(lag_7).alias(
            "demand_rolling_mean_7"
        ),
        pl.mean_horizontal(lag_14).alias(
            "demand_rolling_mean_14"
        ),
    )


# ---------------------------------------------------------------------
# Multi-horizon forecasting dataset
# ---------------------------------------------------------------------

def create_horizon_dataset(
    df: pl.LazyFrame,
    horizon: int,
) -> pl.LazyFrame:
    horizon_df = df.with_columns(
        pl.col("dt")
        .shift(-horizon)
        .over(GROUP_COLUMNS)
        .alias("target_date"),

        pl.col("recovered_daily_demand")
        .shift(-horizon)
        .over(GROUP_COLUMNS)
        .alias("target_daily_demand"),

        pl.col("stock_hour6_22_cnt")
        .shift(-horizon)
        .over(GROUP_COLUMNS)
        .alias("target_stockout_hours"),

        pl.col("holiday_flag")
        .shift(-horizon)
        .over(GROUP_COLUMNS)
        .alias("target_holiday_flag"),
    )

    horizon_df = horizon_df.filter(
        pl.col("target_date").is_not_null()
        & pl.col("target_daily_demand").is_not_null()
    )

    horizon_df = horizon_df.with_columns(
        pl.lit(horizon).alias("forecast_horizon"),
        pl.col("target_date").dt.weekday().alias(
            "target_weekday"
        ),
        pl.col("target_date").dt.month().alias(
            "target_month"
        ),
        (
            pl.col("target_stockout_hours") == 0
        ).alias("target_is_uncensored"),
    )

    columns = (
        ID_COLUMNS
        + [
            "dt",
            "target_date",
            "forecast_horizon",
        ]
        + FORECAST_FEATURES
        + [
            "target_weekday",
            "target_month",
            "target_holiday_flag",
            "target_daily_demand",
            "target_stockout_hours",
            "target_is_uncensored",
        ]
    )

    return horizon_df.select(columns).rename(
        {"dt": "origin_date"}
    )


def build_forecasting_dataset(
    df: pl.LazyFrame,
) -> pl.LazyFrame:
    print("\nBuilding seven forecast horizons...")

    datasets = [
        create_horizon_dataset(df, horizon)
        for horizon in range(1, FORECAST_HORIZON + 1)
    ]

    return pl.concat(
        datasets,
        how="vertical",
    )


# ---------------------------------------------------------------------
# Eligible forecast origins
# ---------------------------------------------------------------------

def get_eligible_origin_dates(
    df: pl.LazyFrame,
) -> list:
    print("\nIdentifying eligible forecast origins...")

    complete_features = pl.all_horizontal(
        [
            pl.col(column).is_not_null()
            for column in FORECAST_FEATURES
        ]
    )

    eligible = (
        df.filter(
            pl.col("forecast_horizon") == FORECAST_HORIZON
        )
        .group_by("origin_date")
        .agg(
            pl.len().alias("total_rows"),
            complete_features.sum().alias("complete_rows"),
        )
        .filter(
            pl.col("total_rows") == pl.col("complete_rows")
        )
        .sort("origin_date")
        .select("origin_date")
        .collect()
    )

    dates = eligible["origin_date"].to_list()

    if not dates:
        raise ValueError(
            "No eligible forecast origins were found."
        )

    for previous, current in zip(dates, dates[1:]):
        if current - previous != timedelta(days=1):
            raise ValueError(
                f"Gap in eligible forecast origins: "
                f"{previous} to {current}"
            )

    print(f"Eligible origins: {len(dates)}")
    print(f"First origin:     {dates[0]}")
    print(f"Last origin:      {dates[-1]}")

    return dates


# ---------------------------------------------------------------------
# Split sizes
# ---------------------------------------------------------------------

def calculate_split_sizes(
    total_origins: int,
) -> tuple[int, int, int]:
    available = total_origins - (2 * EMBARGO_DAYS)

    if available < 3:
        raise ValueError(
            "Insufficient forecast origins for chronological "
            "60/20/20 splits and embargo periods."
        )

    ratios = [
        TRAIN_RATIO,
        VALIDATION_RATIO,
        TEST_RATIO,
    ]

    raw_sizes = [
        available * ratio
        for ratio in ratios
    ]

    sizes = [
        floor(size)
        for size in raw_sizes
    ]

    remaining = available - sum(sizes)

    remainders = [
        raw_sizes[index] - sizes[index]
        for index in range(3)
    ]

    priority = {
        0: 2,
        1: 0,
        2: 1,
    }

    order = sorted(
        range(3),
        key=lambda index: (
            -remainders[index],
            priority[index],
        ),
    )

    for index in order[:remaining]:
        sizes[index] += 1

    if min(sizes) < 1:
        raise ValueError(
            "At least one partition has no forecast origins."
        )

    return tuple(sizes)


# ---------------------------------------------------------------------
# Chronological train / validation / test split
# ---------------------------------------------------------------------

def assign_splits(
    df: pl.LazyFrame,
    eligible_dates: list,
):
    train_size, validation_size, test_size = (
        calculate_split_sizes(len(eligible_dates))
    )

    validation_start_index = (
        train_size + EMBARGO_DAYS
    )

    validation_end_index = (
        validation_start_index + validation_size
    )

    test_start_index = (
        validation_end_index + EMBARGO_DAYS
    )

    train_dates = eligible_dates[:train_size]

    validation_dates = eligible_dates[
        validation_start_index:validation_end_index
    ]

    test_dates = eligible_dates[test_start_index:]

    if len(test_dates) != test_size:
        raise ValueError(
            "Unexpected test partition size."
        )

    validation_start = validation_dates[0]
    test_start = test_dates[0]

    denominator = (
        len(eligible_dates) - 2 * EMBARGO_DAYS
    )

    print("\n" + "=" * 65)
    print("CHRONOLOGICAL 60/20/20 SPLIT")
    print("=" * 65)

    print(
        f"Training:   {len(train_dates):>5} origins "
        f"({len(train_dates) / denominator * 100:.2f}%)"
    )
    print(
        f"Period: {train_dates[0]} to {train_dates[-1]}"
    )

    print(
        f"\nValidation: {len(validation_dates):>5} origins "
        f"({len(validation_dates) / denominator * 100:.2f}%)"
    )
    print(
        f"Period: {validation_dates[0]} to "
        f"{validation_dates[-1]}"
    )

    print(
        f"\nTest:       {len(test_dates):>5} origins "
        f"({len(test_dates) / denominator * 100:.2f}%)"
    )
    print(
        f"Period: {test_dates[0]} to {test_dates[-1]}"
    )

    print(
        f"\nEmbargo between partitions: {EMBARGO_DAYS} days"
    )

    training = df.filter(
        pl.col("origin_date").is_in(train_dates)
        & (pl.col("target_date") < validation_start)
    )

    validation = df.filter(
        pl.col("origin_date").is_in(validation_dates)
        & (pl.col("target_date") < test_start)
    )

    test = df.filter(
        pl.col("origin_date").is_in(test_dates)
    )

    return training, validation, test


# ---------------------------------------------------------------------
# Temporal validation
# ---------------------------------------------------------------------

def validate_temporal_splits(
    training: pl.LazyFrame,
    validation: pl.LazyFrame,
    test: pl.LazyFrame,
) -> None:
    print("\nValidating temporal separation...")

    train_summary = training.select(
        pl.col("target_date").max().alias("max_target")
    ).collect()

    validation_summary = validation.select(
        pl.col("target_date").min().alias("min_target"),
        pl.col("target_date").max().alias("max_target"),
    ).collect()

    test_summary = test.select(
        pl.col("target_date").min().alias("min_target")
    ).collect()

    train_max = train_summary["max_target"][0]
    validation_min = validation_summary["min_target"][0]
    validation_max = validation_summary["max_target"][0]
    test_min = test_summary["min_target"][0]

    if any(
        value is None
        for value in [
            train_max,
            validation_min,
            validation_max,
            test_min,
        ]
    ):
        raise ValueError(
            "One or more forecasting partitions are empty."
        )

    if train_max >= validation_min:
        raise ValueError(
            "Training and validation target periods overlap."
        )

    if validation_max >= test_min:
        raise ValueError(
            "Validation and test target periods overlap."
        )

    print("Training target end:      ", train_max)
    print(
        "Validation target period: ",
        validation_min,
        "to",
        validation_max,
    )
    print("Test target start:        ", test_min)

    print("Temporal separation checks passed.")


# ---------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------

def save_dataset(
    df: pl.LazyFrame,
    filename: str,
) -> None:
    path = OUTPUT_DIR / filename

    print(f"\nSaving {path}...")

    df.sink_parquet(
        path,
        mkdir=True,
    )

    print("Saved successfully.")


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def report_dataset(
    name: str,
    path: Path,
) -> None:
    df = pl.scan_parquet(path)

    summary = df.select(
        pl.len().alias("rows"),
        pl.col("store_id").n_unique().alias("stores"),
        pl.col("product_id").n_unique().alias("products"),
        pl.col("origin_date").min().alias("first_origin"),
        pl.col("origin_date").max().alias("last_origin"),
        pl.col("target_date").min().alias("first_target"),
        pl.col("target_date").max().alias("last_target"),
        pl.col("forecast_horizon").min().alias("min_horizon"),
        pl.col("forecast_horizon").max().alias("max_horizon"),
        pl.col("target_daily_demand").mean().alias("mean_target"),
        pl.col("target_daily_demand").min().alias("min_target"),
        pl.col("target_daily_demand").max().alias("max_target"),
    ).collect()

    print("\n" + "=" * 65)
    print(f"{name.upper()} DATASET")
    print("=" * 65)
    print(summary)

    missingness = df.select(
        pl.len().alias("total_rows"),
        *[
            pl.col(column).null_count().alias(column)
            for column in FORECAST_FEATURES
        ],
    ).collect()

    total_rows = missingness["total_rows"][0]

    print("\nForecast feature missingness:")

    for column in FORECAST_FEATURES:
        missing = missingness[column][0]

        percentage = (
            missing / total_rows * 100
            if total_rows
            else 0
        )

        print(
            f"{column:<32} "
            f"{missing:>12,} "
            f"({percentage:>6.2f}%)"
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("PERISHABLE PRODUCT DEMAND FORECASTING")
    print("DAILY 7-DAY FORECAST DATA PREPARATION")
    print("=" * 65)

    print(
        f"Polars thread pool size: {pl.thread_pool_size()}"
    )

    required_paths = [
        TRAIN_FILE,
        EVAL_FILE,
        RIDGE_ALPHA_1,
        RIDGE_ALPHA_10,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 1. Load source data.
    source_df = load_source_data()

    # 2. Load frozen recovery models.
    model_alpha_1, model_alpha_10 = load_recovery_models()

    # 3. Recover historical daily demand.
    recovered_df = build_recovered_daily_dataset(
        source_df,
        model_alpha_1,
        model_alpha_10,
    )

    recovered_path = OUTPUT_DIR / "recovered_daily_demand.parquet"

    print(f"\nSaving recovered demand: {recovered_path}")

    recovered_df.write_parquet(recovered_path)

    # 4. Create historical forecasting features.
    featured_df = create_features(
        recovered_df.lazy()
    )

    # 5. Build seven forecast horizons.
    forecasting_df = build_forecasting_dataset(
        featured_df
    )

    # 6. Identify eligible forecast origins.
    eligible_dates = get_eligible_origin_dates(
        forecasting_df
    )

    # 7. Create chronological 60/20/20 partitions.
    training, validation, test = assign_splits(
        forecasting_df,
        eligible_dates,
    )

    # 8. Validate temporal separation.
    validate_temporal_splits(
        training,
        validation,
        test,
    )

    # 9. Save partitions.
    save_dataset(training, "train.parquet")
    save_dataset(validation, "validation.parquet")
    save_dataset(test, "test.parquet")

    # 10. Report partitions.
    report_dataset(
        "Training",
        OUTPUT_DIR / "train.parquet",
    )

    report_dataset(
        "Validation",
        OUTPUT_DIR / "validation.parquet",
    )

    report_dataset(
        "Test",
        OUTPUT_DIR / "test.parquet",
    )

    print("\n" + "=" * 65)
    print("DAILY 7-DAY FORECAST DATA PREPARATION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
