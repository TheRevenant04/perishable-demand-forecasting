# Machine Learning Pipeline

The workflow has two modelling stages: **latent demand recovery** (preprocessing)
and **seven-day demand forecasting** (the deployed model).

```text
Raw data -> inspection / EDA -> recovery dataset -> recovery models
         -> recovered daily demand -> forecasting dataset
         -> forecasting models -> final model -> API
```

## 1. Inspection and exploratory analysis

Validate the raw schema, date coverage, nulls, duplicates, hourly arrays and
sales reconciliation, then study stockout patterns.

```bash
uv run python -m perishable_demand_forecasting.analysis.inspect_raw_dataset
uv run python -m perishable_demand_forecasting.analysis.analyze_hourly_demand_patterns
uv run python -m perishable_demand_forecasting.analysis.analyze_matched_stockouts
uv run python -m perishable_demand_forecasting.analysis.analyze_stockouts
```

Outputs are diagnostics only; no data is transformed.

## 2. Build the recovery dataset

Fully in-stock observations become reconstruction targets. Stockout masks from
other observations are applied to their hourly sales to simulate missing demand.
Chronological train/validation/test recovery splits are written to
`data/processed/freshretailnet/recovery/`.

```bash
uv run python -m perishable_demand_forecasting.recovery.build_recovery_dataset
uv run python -m perishable_demand_forecasting.recovery.evaluate_recovery_baseline
```

## 3. Recover latent demand

Calendar, product, business, weather, management and hourly-sales features are
prepared, then grouped into experiments A–H to compare feature sets.

```bash
uv run python -m perishable_demand_forecasting.recovery.build_recovery_features
uv run python -m perishable_demand_forecasting.recovery.inspect_recovery_features
uv run python -m perishable_demand_forecasting.recovery.select_recovery_features
uv run python -m perishable_demand_forecasting.recovery.train_recovery_experiments
```

Experiment **G (business + hourly)** is selected on validation performance. Ridge
`alpha` values are then compared, and a hybrid routing strategy is frozen:

| Artificial stockout severity | Strategy |
| --- | --- |
| 0–5 hours | Use observed daily sales |
| 6–9 hours | Ridge `alpha=10` |
| 10–16 hours | Ridge `alpha=1` |

```bash
uv run python -m perishable_demand_forecasting.recovery.tune_recovery_ridge
uv run python -m perishable_demand_forecasting.recovery.evaluate_recovery_by_stockout_severity
uv run python -m perishable_demand_forecasting.recovery.evaluate_hybrid_strategies
uv run python -m perishable_demand_forecasting.recovery.test_hybrid_robustness
uv run python -m perishable_demand_forecasting.recovery.evaluate_hybrid_on_test
```

Negative recovered demand is clipped to zero before forecasting features are
built. See [results.md](results.md) for the recovery evaluation.

## 4. Build the forecasting dataset

The recovered daily demand series is expanded into supervised forecasting rows:

- demand lags 1–14
- 7-day and 14-day rolling mean demand
- lagged stockout hours (1 and 7 days)
- forecast horizon 1–7
- future-known calendar fields: weekday, month, holiday flag

Runs are split chronologically with a seven-day embargo.

```bash
uv run python -m perishable_demand_forecasting.data.build_daily_forecast_dataset
```

Outputs: `recovered_daily_demand.parquet`, `train.parquet`, `validation.parquet`
and `test.parquet` under `data/processed/freshretailnet/forecasting/`.

## 5. Forecast demand

| Model | Role |
| --- | --- |
| Naive | Predicts every horizon with `demand_lag_1`. |
| Ridge Regression | Uses the 22 lag, rolling, stockout, horizon and calendar features. |

The scaler is fitted on training data only. Ridge `alpha` is tuned on validation;
differences were negligible, so `alpha=1.0` is retained.

```bash
uv run python -m perishable_demand_forecasting.forecasting.evaluate_naive_forecast
uv run python -m perishable_demand_forecasting.forecasting.train_ridge_forecast
uv run python -m perishable_demand_forecasting.forecasting.tune_ridge_forecast
uv run python -m perishable_demand_forecasting.forecasting.evaluate_ridge_forecast_test
```

## 6. Train the final model

`train.py` refits the selected configuration on train + validation and saves the
bundle used by the service.

```bash
uv run python train.py
```

Details: [results.md](results.md) and [serving.md](serving.md).
