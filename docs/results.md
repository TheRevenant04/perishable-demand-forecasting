# Results and Evaluation

## Evaluation protocol

- Chronological train, validation and test splits.
- The test partition is untouched during feature selection and tuning.
- Forecasting partitions use a seven-day embargo to reduce temporal leakage.
- Metrics: MAE, RMSE and bias, overall and by forecast horizon.

## Demand recovery

Recovery is evaluated on an artificial-mask test set, where the original daily
demand is known because fully in-stock observations are masked on purpose. This
is a methodological proxy, not verified real-stockout ground truth.

| Method | MAE | RMSE |
| --- | ---: | ---: |
| Observed-sales baseline | 0.418767 | 0.977165 |
| Frozen hybrid recovery | **0.259736** | **0.587402** |

The hybrid reduced MAE by approximately **38%** relative to the observed-sales
baseline. Negative recovered values are clipped to zero before forecasting
features are built.

## Seven-day forecasting

Final evaluation on the held-out test partition:

| Metric | Naive | Ridge Regression |
| --- | ---: | ---: |
| MAE | 0.512812 | **0.407412** |
| RMSE | 0.814507 | **0.648432** |
| Bias | -0.037006 | -0.114638 |

Ridge achieved approximately **20.55% lower MAE** and **20.39% lower RMSE** than
the Naive baseline, at the cost of a more negative bias.

## Feature importance

Ridge coefficients on standardized inputs show where the forecast comes from:

| Feature group | Share of absolute coefficient mass |
| --- | ---: |
| Demand lags (1–14 days) | ~75.0% |
| Rolling demand (7- and 14-day means) | ~14.5% |
| Calendar fields | ~9.3% |
| Stockout history | ~1.2% |
| Forecast horizon | ~0.05% |

Recent demand and rolling means dominate, which is consistent with the strong
autocorrelation of daily demand. Horizon contributes almost nothing, indicating
the model forecasts from the historical level rather than horizon-specific
behaviour.

## Limitations

- Artificial masking is a proxy; real lost sales remain unobserved.
- Ridge produces a small share of negative predictions (3.54% on test).
- Bias is more negative than the Naive baseline.
- The approach is evaluated on one dataset and a short test period.
