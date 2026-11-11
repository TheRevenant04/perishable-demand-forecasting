# Model Serving

The serving layer is separate from the ML workflow. It loads a trained artifact
and exposes it over HTTP; it never imports training code.

For the complete endpoint reference, request schemas, error codes and example
payloads, see [api.md](api.md) and [`examples/`](../examples).

## Train the artifact

```bash
uv run python train.py
```

`train.py` fits `StandardScaler` + `Ridge(alpha=1.0)` on the train and validation
splits and saves `models/forecasting/final_demand_forecast.pkl`. The bundle
contains the model, scaler, feature order, target name, alpha, timestamp and row
count.

## Start the service

```bash
uv run python predict.py
```

Default address: `http://localhost:8000`. Host and port can be changed with the
`HOST` and `PORT` environment variables.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Service metadata and required feature list. |
| `GET` | `/health` | Model status; `503` if the artifact is missing. |
| `POST` | `/predict` | Forecast one or many instances. |

### Request formats

Single instance:

```json
{"features": {"demand_lag_1": 5.0, "...": "..."}}
```

Batch:

```json
{"instances": [{"demand_lag_1": 5.0, "...": "..."}, {"demand_lag_1": 3.0, "...": "..."}]}
```

### Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {
    "demand_lag_1": 5, "demand_lag_2": 4, "demand_lag_3": 4,
    "demand_lag_4": 3, "demand_lag_5": 3, "demand_lag_6": 3,
    "demand_lag_7": 3, "demand_lag_8": 2, "demand_lag_9": 2,
    "demand_lag_10": 2, "demand_lag_11": 2, "demand_lag_12": 2,
    "demand_lag_13": 2, "demand_lag_14": 2,
    "demand_rolling_mean_7": 3.4, "demand_rolling_mean_14": 2.9,
    "stockout_hours_lag_1": 0, "stockout_hours_lag_7": 0,
    "forecast_horizon": 1, "target_weekday": 2,
    "target_month": 5, "target_holiday_flag": 0
  }}'
```

Response:

```json
{"prediction": 3.2214, "target": "target_daily_demand"}
```

Batch responses return `predictions` plus a `count`. Invalid JSON, missing
features, non-numeric or non-finite values return HTTP `400`.

## Required features

`demand_lag_1` … `demand_lag_14`, `demand_rolling_mean_7`,
`demand_rolling_mean_14`, `stockout_hours_lag_1`, `stockout_hours_lag_7`,
`forecast_horizon`, `target_weekday`, `target_month`, `target_holiday_flag`.

## Docker

```bash
uv run python train.py
docker build -t perishable-demand-forecasting .
docker run --rm -p 8000:8000 perishable-demand-forecasting
```

The image installs runtime dependencies with `uv`, runs as a non-root user, and
defines a healthcheck against `/health`. Build context excludes `data/` and
`.venv/` via `.dockerignore`, so the image only needs the model artifact.

| Variable | Default in image | Purpose |
| --- | --- | --- |
| `PERISHABLE_MODELS_DIR` | `/app/models` | Model artifact location. |
| `HOST` | `0.0.0.0` | Bind address. |
| `PORT` | `8000` | Listening port. |
