# Project Structure

The repository separates the offline ML workflow from the online serving layer.
Generated data and models are kept out of version control.

```text
perishable-demand-forecasting/
├── train.py                        # train the final model
├── predict.py                      # start the forecast web service
├── Dockerfile                      # container image for the service
├── pyproject.toml                  # dependencies and package configuration
├── uv.lock                         # pinned dependency lockfile
├── README.md                       # problem, solution and documentation map
├── docs/                           # detailed documentation
├── notebooks/
│   └── notebook.ipynb              # end-to-end narrative walkthrough
├── src/
│   └── perishable_demand_forecasting/
│       ├── config.py               # central paths and environment overrides
│       ├── data/                   # download and dataset construction
│       ├── analysis/               # inspection, EDA and stockout analysis
│       ├── recovery/               # latent demand recovery
│       ├── forecasting/            # forecasting models and evaluation
│       └── api/                    # model loading and Flask routes
├── data/                           # raw and processed data (gitignored)
├── models/                         # trained artifacts (gitignored)
└── tests/                          # test scaffold
```

## Root entrypoints

| File | Purpose |
| --- | --- |
| `train.py` | Fits the final Random Forest on train + validation and saves a pickle file. |
| `predict.py` | Starts the Flask service that loads the saved bundle. |
| `Dockerfile` | Multi-stage image that installs runtime dependencies and serves the model. |

## Package directories

| Directory | Contents |
| --- | --- |
| `config.py` | Project paths plus `PERISHABLE_DATA_DIR` / `PERISHABLE_MODELS_DIR` overrides. |
| `data/` | `download_raw_data.py` fetches the dataset; `build_daily_forecast_dataset.py` builds lag/horizon features and splits. |
| `analysis/` | Raw-data validation, hourly demand profiling, matched-stockout comparison and stockout analysis. |
| `recovery/` | Builds the artificial-mask recovery dataset, feature groups, Ridge experiments, tuning and hybrid evaluation. |
| `forecasting/` | Naive baseline, Ridge training and tuning, tree-based tuning (decision tree, random forest, XGBoost), and test evaluation. |
| `api/` | `model.py` loads the artifact and predicts; `app.py` defines the Flask routes. |

## Generated directories

| Directory | Contents |
| --- | --- |
| `data/raw/freshretailnet/` | Downloaded `train.parquet` and `eval.parquet`. |
| `data/processed/freshretailnet/features/` | Model-ready recovery features and A–H experiment subsets. |
| `data/processed/freshretailnet/recovery/` | Recovery splits, metrics and severity evaluations. |
| `data/processed/freshretailnet/forecasting/` | Forecasting splits, lag features and metrics. |
| `models/recovery/` | Recovery experiment and regularization artifacts. |
| `models/forecasting/` | Validation candidates (Ridge, tuned tree) and the final Random Forest `final_demand_forecast.pkl`. |

## Conventions

- Modules are runnable with `python -m`, e.g.
  `uv run python -m perishable_demand_forecasting.analysis.inspect_raw_dataset`.
- Data-processing modules read and write the paths defined in `config.py`, so
  they can be invoked from any working directory.
- The serving layer never imports training code; the pickle artifact is the
  boundary between the two halves of the project.
