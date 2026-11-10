from pathlib import Path

from huggingface_hub import hf_hub_download

DATASET_ID = "Dingdong-Inc/FreshRetailNet-50K"
DATA_DIR = Path("data/raw/freshretailnet")


def download_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    files = [
        "data/train.parquet",
        "data/eval.parquet",
    ]

    for filename in files:
        path = hf_hub_download(
            repo_id=DATASET_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=DATA_DIR,
        )

        print(f"Downloaded: {path}")


if __name__ == "__main__":
    download_dataset()