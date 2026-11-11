# Forecast API Reference

The service exposes the trained Random Forest model over HTTP using Flask. It
loads `models/forecasting/final_demand_forecast.pkl` once per process.

Start it with:

```bash
uv run python train.py      # creates the artifact if missing
uv run python predict.py
```

Default base URL: `http://localhost:8000`. Configure with `HOST` and `PORT`.

There is no authentication; deploy behind your own gateway for production use.
Requests and responses use JSON (`Content-Type: application/json`).

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Service metadata, model info and required features. |
| `GET` | `/health` | Liveness/model status. |
| `POST` | `/predict` | Forecast one or many instances. |

## `GET /`

```json
{
  "service": "perishable-demand-forecasting",
  "model": "random_forest",
  "params": {"n_estimators": 60, "max_depth": 14, "min_samples_leaf": 20},
  "trained_at": "2026-11-11T13:36:16+00:00",
  "features": ["demand_lag_1", "..."],
  "endpoints": {
    "GET /health": "service status",
    "POST /predict": "JSON body: {\"instances\": [{...}]}"
  }
}
```

## `GET /health`

Returns `200` when the artifact loads, otherwise `503`.

```json
{"status": "ok", "model": "random_forest", "params": {"n_estimators": 60}, "trained_at": "..."}
```

```json
{"status": "error", "detail": "Trained model not found: ..."}
```

## `POST /predict`

### Request body

Single instance:

```json
{"features": {"demand_lag_1": 5.0, "...": "..."}}
```

Batch:

```json
{"instances": [{"demand_lag_1": 5.0, "...": "..."}, {"demand_lag_1": 3.0, "...": "..."}]}
```

### Required features

| Group | Fields |
| --- | --- |
| Demand lags | `demand_lag_1` … `demand_lag_14` |
| Rolling demand | `demand_rolling_mean_7`, `demand_rolling_mean_14` |
| Stockout history | `stockout_hours_lag_1`, `stockout_hours_lag_7` |
| Horizon | `forecast_horizon` (1–7) |
| Calendar | `target_weekday`, `target_month`, `target_holiday_flag` |

Every instance must include all 22 fields with finite numeric values. Tree
models consume the values directly; a legacy linear bundle applies its saved
scaler before prediction.

### Success responses

Single (`200`):

```json
{"prediction": 2.4694, "target": "target_daily_demand"}
```

Batch (`200`):

```json
{
  "predictions": [2.4694, 2.2988, 1.5404, 1.6591, 1.1537],
  "count": 5,
  "target": "target_daily_demand"
}
```

### Error responses

| Status | Condition | Body |
| --- | --- | --- |
| `400` | Body is not valid JSON | `{"error": "Request body must be valid JSON."}` |
| `400` | Body lacks `features`/`instances`, or `instances` is empty/not a list | `{"error": "..."}` |
| `400` | Instance is not an object or is missing fields | `{"error": "Instance 0 is missing features: ..."}` |
| `400` | Non-numeric or non-finite values | `{"error": "..."}` |
| `503` | `/health` when the model artifact is absent | `{"status": "error", "detail": "..."}` |

## Example test data

Ready-to-use payloads live in [`examples/`](../examples):

| File | Contents |
| --- | --- |
| `examples/predict_single_request.json` | One real instance from the held-out forecasting test split. |
| `examples/predict_batch_request.json` | Five real test instances for batch prediction. |

These were extracted from `data/processed/freshretailnet/forecasting/test.parquet`,
the chronological test partition produced by the pipeline. That file is
gitignored; regenerate it with
`uv run python -m perishable_demand_forecasting.data.build_daily_forecast_dataset`.

### Test with curl

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  --data @examples/predict_single_request.json
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  --data @examples/predict_batch_request.json
```

### Test with Python

```python
import json
import urllib.request

payload = json.load(open("examples/predict_single_request.json"))
request = urllib.request.Request(
    "http://localhost:8000/predict",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
print(json.load(urllib.request.urlopen(request)))
```

## Docker

```bash
docker build -t perishable-demand-forecasting .
docker run --rm -p 8000:8000 perishable-demand-forecasting
```

The container uses the same endpoints and reads the model from
`/app/models` (`PERISHABLE_MODELS_DIR`). See [serving.md](serving.md) for the
artifact and deployment details.
