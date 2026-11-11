
from pathlib import Path

import numpy as np
import polars as pl

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATA_FILE = Path("data/raw/freshretailnet/data/train.parquet")

OUTPUT_DIR = Path("data/processed/freshretailnet/recovery")

# Preserve the chronological split boundaries.
SPLITS = {
    "train": {
        "start": "2024-04-11",
        "end": "2024-05-17",
        "seed": 42,
        "samples": 180_000,
    },
    "validation": {
        "start": "2024-05-25",
        "end": "2024-06-06",
        "seed": 43,
        "samples": 60_000,
    },
    "test": {
        "start": "2024-06-14",
        "end": "2024-06-25",
        "seed": 44,
        "samples": 60_000,
    },
}

SALES_COLUMN = "sale_amount"
STOCKOUT_COLUMN = "stock_hour6_22_cnt"

HOURLY_SALES_COLUMN = "hours_sale"
HOURLY_STATUS_COLUMN = "hours_stock_status"

OPERATING_START_HOUR = 6
OPERATING_END_HOUR = 21

OPERATING_HOURS = np.arange(
    OPERATING_START_HOUR,
    OPERATING_END_HOUR + 1,
)

TOTAL_SAMPLES = sum(
    configuration["samples"]
    for configuration in SPLITS.values()
)


# ---------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------

def load_data() -> pl.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {DATA_FILE}"
        )

    print(f"Loading dataset: {DATA_FILE}")

    df = pl.read_parquet(DATA_FILE)

    required_columns = [
        "dt",
        "store_id",
        "product_id",
        SALES_COLUMN,
        STOCKOUT_COLUMN,
        HOURLY_SALES_COLUMN,
        HOURLY_STATUS_COLUMN,
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df


def validate_hourly_arrays(df: pl.DataFrame) -> None:
    print("\nValidating hourly arrays...")

    invalid_lengths = df.filter(
        (pl.col(HOURLY_SALES_COLUMN).list.len() != 24)
        | (pl.col(HOURLY_STATUS_COLUMN).list.len() != 24)
    ).height

    if invalid_lengths:
        raise ValueError(
            f"Found {invalid_lengths:,} rows without 24 hourly values."
        )

    invalid_stockout_hours = df.filter(
        ~pl.col(STOCKOUT_COLUMN).is_between(
            pl.lit(0),
            pl.lit(16),
        )
    ).height

    if invalid_stockout_hours:
        raise ValueError(
            f"Found {invalid_stockout_hours:,} invalid stockout-hour values."
        )

    print("Hourly array lengths and stockout-hour values are valid.")


# ---------------------------------------------------------------------
# Split selection
# ---------------------------------------------------------------------

def select_split(
    df: pl.DataFrame,
    split_name: str,
    configuration: dict,
) -> tuple[pl.DataFrame, pl.DataFrame]:

    start_date = configuration["start"]
    end_date = configuration["end"]

    requested_samples = configuration["samples"]

    split_df = df.filter(
        (pl.col("dt") >= pl.lit(start_date))
        & (pl.col("dt") <= pl.lit(end_date))
    )

    complete_days = split_df.filter(
        pl.col(STOCKOUT_COLUMN) == 0
    )

    mask_donors = split_df.filter(
        pl.col(STOCKOUT_COLUMN) > 0
    )

    print(f"\n{split_name.upper()} SPLIT")
    print("-" * 65)
    print(f"Date range: {start_date} to {end_date}")
    print(f"Total daily observations: {split_df.height:,}")
    print(f"Fully in-stock candidate days: {complete_days.height:,}")
    print(f"Stockout mask donor days: {mask_donors.height:,}")
    print(f"Requested synthetic samples: {requested_samples:,}")

    if complete_days.height < requested_samples:
        raise ValueError(
            f"Not enough fully in-stock observations for {split_name}. "
            f"Required: {requested_samples:,}; "
            f"available: {complete_days.height:,}."
        )

    if mask_donors.is_empty():
        raise ValueError(
            f"No stockout mask donors available for {split_name}."
        )

    return complete_days, mask_donors


# ---------------------------------------------------------------------
# Synthetic daily recovery dataset
# ---------------------------------------------------------------------

def create_recovery_samples(
    complete_days: pl.DataFrame,
    mask_donors: pl.DataFrame,
    split_name: str,
    seed: int,
    sample_count: int,
) -> pl.DataFrame:

    print(f"\nCreating {sample_count:,} synthetic daily samples...")

    sampled_days = complete_days.sample(
        n=sample_count,
        with_replacement=False,
        seed=seed,
    )

    sampled_donors = mask_donors.sample(
        n=sample_count,
        with_replacement=True,
        seed=seed + 100,
    )

    complete_hourly_sales = np.asarray(
        sampled_days[HOURLY_SALES_COLUMN].to_list(),
        dtype=np.float64,
    )

    donor_hourly_status = np.asarray(
        sampled_donors[HOURLY_STATUS_COLUMN].to_list(),
        dtype=np.int8,
    )

    if complete_hourly_sales.shape != (sample_count, 24):
        raise ValueError(
            "Unexpected complete hourly sales matrix shape: "
            f"{complete_hourly_sales.shape}"
        )

    if donor_hourly_status.shape != (sample_count, 24):
        raise ValueError(
            "Unexpected donor hourly status matrix shape: "
            f"{donor_hourly_status.shape}"
        )

    # Apply actual stockout patterns to fully in-stock daily sales.
    simulated_stockout_mask = (
        donor_hourly_status[:, OPERATING_HOURS] == 1
    )

    simulated_hourly_sales = complete_hourly_sales.copy()

    operating_sales = simulated_hourly_sales[:, OPERATING_HOURS]

    operating_sales[simulated_stockout_mask] = 0.0

    simulated_hourly_sales[:, OPERATING_HOURS] = operating_sales

    # One simulated daily observation per store-product-day.
    simulated_daily_sales = simulated_hourly_sales.sum(axis=1)

    simulated_stockout_hours = simulated_stockout_mask.sum(axis=1)

    simulated_stockout_fraction = (
        simulated_stockout_hours / len(OPERATING_HOURS)
    )

    # Original complete daily sales are the known recovery targets.
    target_daily_demand = sampled_days[SALES_COLUMN].to_numpy()

    excluded_columns = {
        HOURLY_SALES_COLUMN,
        HOURLY_STATUS_COLUMN,
        STOCKOUT_COLUMN,
        SALES_COLUMN,
    }

    metadata_columns = [
        column
        for column in sampled_days.columns
        if column not in excluded_columns
    ]

    output_data = sampled_days.select(
        metadata_columns
    ).to_dict(as_series=False)

    output_data["target_daily_demand"] = target_daily_demand

    output_data["simulated_observed_daily_sales"] = (
        simulated_daily_sales
    )

    output_data["simulated_stockout_hours"] = (
        simulated_stockout_hours
    )

    output_data["simulated_stockout_fraction"] = (
        simulated_stockout_fraction
    )

    # Supporting hourly sales features for one daily prediction.
    for hour in range(24):
        output_data[f"observed_hour_{hour:02d}"] = (
            simulated_hourly_sales[:, hour]
        )

    output_data["split"] = [split_name] * sample_count

    result = pl.DataFrame(output_data)

    # Basic validation.
    if result["target_daily_demand"].null_count() > 0:
        raise ValueError("Recovery targets contain null values.")

    if result["simulated_observed_daily_sales"].null_count() > 0:
        raise ValueError("Simulated daily sales contain null values.")

    if (
        result["simulated_observed_daily_sales"]
        > result["target_daily_demand"] + 1e-8
    ).any():
        raise ValueError(
            "Simulated observed daily sales exceed the original target."
        )

    print(f"Generated samples: {result.height:,}")

    print(
        "Average original daily demand: "
        f"{result['target_daily_demand'].mean():.6f}"
    )

    print(
        "Average simulated observed daily sales: "
        f"{result['simulated_observed_daily_sales'].mean():.6f}"
    )

    print(
        "Average simulated stockout hours: "
        f"{result['simulated_stockout_hours'].mean():.2f}"
    )

    print(
        "Fully masked daily samples: "
        f"{(result['simulated_stockout_hours'] == 16).sum():,}"
    )

    return result


# ---------------------------------------------------------------------
# Save datasets
# ---------------------------------------------------------------------

def save_dataset(
    df: pl.DataFrame,
    split_name: str,
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = OUTPUT_DIR / f"{split_name}.parquet"

    df.write_parquet(
        output_path,
        compression="zstd",
    )

    print(f"Saved: {output_path}")
    print(f"Rows: {df.height:,}")
    print(f"Columns: {df.width:,}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print("=" * 75)
    print("DAILY DEMAND RECOVERY DATASET PREPARATION")
    print("=" * 75)

    print("\nTarget sample distribution:")

    for split_name, configuration in SPLITS.items():
        percentage = (
            configuration["samples"] / TOTAL_SAMPLES
        ) * 100

        print(
            f"{split_name.capitalize():<12}"
            f"{configuration['samples']:>12,}"
            f"{percentage:>10.2f}%"
        )

    print("-" * 35)
    print(f"{'Total':<12}{TOTAL_SAMPLES:>12,}{100:>10.2f}%")

    df = load_data()

    validate_hourly_arrays(df)

    for split_name, configuration in SPLITS.items():

        complete_days, mask_donors = select_split(
            df,
            split_name,
            configuration,
        )

        recovery_samples = create_recovery_samples(
            complete_days=complete_days,
            mask_donors=mask_donors,
            split_name=split_name,
            seed=configuration["seed"],
            sample_count=configuration["samples"],
        )

        save_dataset(
            recovery_samples,
            split_name,
        )

    print("\n" + "=" * 75)
    print("DAILY RECOVERY DATASET PREPARATION COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()
