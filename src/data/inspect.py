from pathlib import Path

import polars as pl

DATA_DIR = Path("data/raw/freshretailnet")

FILES = {
    "Train": DATA_DIR / "data/train.parquet",
    "Evaluation": DATA_DIR / "data/eval.parquet",
}


def inspect_file(name: str, path: Path):
    if not path.exists():
        print(f"\n{name}: File not found at {path}")
        return

    print(f"\n{'=' * 60}")
    print(f"{name} Dataset")
    print(f"{'=' * 60}")

    df = pl.scan_parquet(path)

    # Dataset schema
    print("\nSchema:")
    print(df.collect_schema())

    # Dataset dimensions
    row_count = df.select(pl.len()).collect().item()
    column_count = len(df.collect_schema())

    print(f"\nRows: {row_count:,}")
    print(f"Columns: {column_count}")

    # Sample records
    print("\nFirst 5 records:")
    print(df.head(5).collect())


def main():
    for name, path in FILES.items():
        inspect_file(name, path)


if __name__ == "__main__":
    main()