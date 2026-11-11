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

Model selection on the validation partition:

| Model | Tuned settings | MAE | RMSE | Bias |
| --- | --- | ---: | ---: | ---: |
| Naive (lag 1) | — | 0.477620 | 0.803075 | 0.020007 |
| Ridge Regression | alpha=1.0 | 0.374763 | 0.604713 | 0.001554 |
| Decision Tree | depth=12, leaf=100 | 0.368541 | 0.602604 | 0.032915 |
| XGBoost | lr=0.05, depth=10, early stopping | 0.368517 | 0.640420 | 0.022574 |
| Random Forest | depth=14, leaf=20, 60 trees | **0.361513** | **0.577953** | 0.031059 |

Decision trees, random forests and XGBoost are the tree-based models covered in
ML Zoomcamp module 6. Their main regularization parameters are tuned on
deterministic training samples and every candidate is scored on all 4.55M
validation rows. Random Forest wins on validation MAE and is selected as the
final model.

Final evaluation on the untouched test partition:

| Model | MAE | RMSE | Bias | Negative predictions |
| --- | ---: | ---: | ---: | ---: |
| Naive baseline | 0.512812 | 0.814507 | -0.037006 | 0.00% |
| Ridge Regression | 0.407412 | **0.648432** | -0.114638 | 3.54% |
| Random Forest (final) | **0.399386** | 0.651719 | -0.107226 | 0.00% |

The Random Forest achieved approximately **22.12% lower MAE** than the Naive
baseline and **1.97% lower MAE** than Ridge. Ridge retains a slightly lower RMSE.
Both models have a negative bias; the forest avoids negative predictions but
still under-forecasts on average.

## Feature importance

Random Forest impurity-based importances show where the forecast comes from:

| Feature group | Share of total importance |
| --- | ---: |
| Rolling demand (7- and 14-day means) | ~71.7% |
| Demand lags (1–14 days) | ~24.7% |
| Calendar fields | ~3.3% |
| Stockout history | ~0.2% |
| Forecast horizon | ~0.1% |

The rolling means and recent demand dominate, which is consistent with the
strong autocorrelation of daily demand. Horizon contributes almost nothing,
indicating the model forecasts from the historical level rather than
horizon-specific behaviour.

## Limitations

- Artificial masking is a proxy; real lost sales remain unobserved.
- The forest produces no negative predictions, but it under-forecasts on average.
- Bias is more negative than the Naive baseline and similar to Ridge.
- The approach is evaluated on one dataset and a short test period.
