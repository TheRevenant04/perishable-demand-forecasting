# Setup and Run

This guide covers installing the project, downloading the dataset, running the
pipeline, and starting the forecast service.

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (dependency and environment manager)
- Docker (optional, for the containerised service)

## Install dependencies

From the repository root:

```bash
uv sync
```

This creates `.venv` and installs the package in editable mode, including the
development tools (`ipykernel`, `jupyter`). Runtime dependencies are declared in
[`pyproject.toml`](../pyproject.toml) and pinned in [`uv.lock`](../uv.lock).

## Download the dataset

```bash
uv run python -m perishable_demand_forecasting.data.download_raw_data
```

The FreshRetailNet-50K files are downloaded to
`data/raw/freshretailnet/data/train.parquet` and `eval.parquet`.

## Run the workflow

The full pipeline is documented in [pipeline.md](pipeline.md). A typical run
downloads the data, builds the recovery and forecasting datasets, trains the
models, and evaluates them:

```bash
uv run python -m perishable_demand_forecasting.recovery.build_recovery_dataset
uv run python -m perishable_demand_forecasting.recovery.build_recovery_features
uv run python -m perishable_demand_forecasting.recovery.select_recovery_features
uv run python -m perishable_demand_forecasting.recovery.train_recovery_experiments
uv run python -m perishable_demand_forecasting.recovery.tune_recovery_ridge
uv run python -m perishable_demand_forecasting.data.build_daily_forecast_dataset
uv run python -m perishable_demand_forecasting.forecasting.train_ridge_forecast
uv run python -m perishable_demand_forecasting.forecasting.tune_ridge_forecast
uv run python -m perishable_demand_forecasting.forecasting.evaluate_ridge_forecast_test
```

## Train the final model

```bash
uv run python train.py
```

This fits the final Ridge model on the train + validation splits and writes
`models/forecasting/final_demand_forecast.joblib`.

## Start the forecast service

```bash
uv run python predict.py
```

The API listens on `http://localhost:8000`. See [api.md](api.md) for the endpoint
reference and [serving.md](serving.md) for deployment details.

Test it with a committed sample payload from the held-out test split:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  --data @examples/predict_single_request.json
```

## Run with Docker

The model must exist before building the image:

```bash
uv run python train.py
docker build -t perishable-demand-forecasting .
docker run --rm -p 8000:8000 perishable-demand-forecasting
```

## Run the notebook

```bash
uv run jupyter notebook notebooks/notebook.ipynb
```

See [notebook.md](notebook.md) for what the notebook covers.

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `PERISHABLE_DATA_DIR` | `./data` | Overrides the data directory. |
| `PERISHABLE_MODELS_DIR` | `./models` | Overrides the model artifact directory. |
| `HOST` | `0.0.0.0` | Bind address for the API. |
| `PORT` | `8000` | Port for the API. |
