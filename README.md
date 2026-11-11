# Perishable Demand Forecasting

Estimating the demand customers actually had — not just the sales a store
managed to record — and forecasting the next seven days for every store-product.

## The Problem

Managing perishable inventory is a balancing act. Order too little and products
sell out, losing sales and availability. Order too much and stock expires,
creating markdowns and waste.

The data makes this harder, because **recorded sales are not the same as demand**.
If a product sells 50 units before the shelf empties but 70 customers wanted it,
the missing 20 units never appear in the sales history.

A model trained naively on sales learns the wrong lesson:

```text
Low recorded sales -> low forecast -> insufficient stock -> stockout -> even fewer sales
```

For perishable products this matters twice over: retailers must balance
**availability**, **inventory cost** and **waste** at the same time.

More on the data and its stockout structure: [data.md](docs/data.md).

## The Solution

The project treats forecasting as a two-stage problem:

1. **Recover latent demand.** Detect stockout-affected history and estimate what
   demand would have been if the product had stayed available.
2. **Forecast future demand.** Use recovered demand history plus lag, calendar
   and stockout features to predict daily demand for horizons 1–7.

The final Random Forest model (selected over Ridge and XGBoost) is packaged as
a reproducible training script and served through a Flask API, so forecasts can
feed replenishment and planning workflows. On the held-out test period it
reduced MAE by **~22.1%** versus a naive baseline.

```text
Raw retail data -> recovery -> recovered demand -> forecasting -> final model -> API
```

How each stage works: [pipeline.md](docs/pipeline.md).

## Documentation Map

| Section | What it explains | Details |
| --- | --- | --- |
| Setup and run | Install with `uv`, download data, train, serve, Docker and environment variables. | [setup.md](docs/setup.md) |
| Project structure | Root entrypoints, package directories, generated artifacts and conventions. | [project-structure.md](docs/project-structure.md) |
| Data | FreshRetailNet-50K, characteristics, splits, artificial masking and license. | [data.md](docs/data.md) |
| ML pipeline | Recovery experiments, hybrid strategy, forecasting features and model tuning. | [pipeline.md](docs/pipeline.md) |
| Notebook | Guided end-to-end walkthrough of data, experiments and evaluation. | [notebook.md](docs/notebook.md) |
| Results and evaluation | Metrics, recovery and forecasting results, feature importance and limitations. | [results.md](docs/results.md) |
| API reference | Endpoints, request schemas, error responses and ready-to-use test payloads. | [api.md](docs/api.md) |
| Serving and deployment | `train.py`, `predict.py`, API contract, request examples and Docker image. | [serving.md](docs/serving.md) |

## Results at a Glance

| Stage | Result |
| --- | --- |
| Recovery (artificial-mask test) | Hybrid MAE **0.2597** vs observed-sales **0.4188** (~38% lower) |
| Forecasting (held-out test) | Random Forest MAE **0.3994** vs Naive **0.5128** (~22.1% lower); Ridge **0.4074** |

See [results.md](docs/results.md) for the full tables and limitations.

## Quick Start

```bash
uv sync
uv run python -m perishable_demand_forecasting.data.download_raw_data

# Build the recovery models and forecasting datasets used by train.py
uv run python -m perishable_demand_forecasting.recovery.build_recovery_dataset
uv run python -m perishable_demand_forecasting.recovery.build_recovery_features
uv run python -m perishable_demand_forecasting.recovery.select_recovery_features
uv run python -m perishable_demand_forecasting.recovery.tune_recovery_ridge
uv run python -m perishable_demand_forecasting.data.build_daily_forecast_dataset

uv run python train.py
uv run python predict.py
```

`uv sync` creates `.venv`; activation is optional because every command uses
`uv run`, but you can activate it manually:

```bash
source .venv/bin/activate    # macOS / Linux
.venv\Scripts\activate       # Windows
```

The dataset build can take several minutes. The complete pipeline (including
evaluation), Docker usage and environment variables are in
[setup.md](docs/setup.md).

To review the end-to-end narrative, open the notebook:

```bash
uv run jupyter notebook notebooks/notebook.ipynb
```

## Docker

Build the image after training the model, then run the service:

```bash
uv run python train.py
docker build -t perishable-demand-forecasting .
docker run --rm -p 8000:8000 perishable-demand-forecasting
```

Check the container:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  --data @examples/predict_single_request.json
```

The image installs dependencies with `uv`, runs as a non-root user, ships the
artifact from `models/forecasting/final_demand_forecast.pkl`, and exposes port
`8000`. See [serving.md](docs/serving.md) for details.

## License

FreshRetailNet-50K is released under **CC BY 4.0**. Review the dataset's terms
before commercial use.
