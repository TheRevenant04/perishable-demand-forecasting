# Notebook Walkthrough

`notebooks/notebook.ipynb` is the end-to-end narrative for the project. It
orchestrates the packaged modules so the data, methodology, experiments and
results can be reviewed in order.

## Run it

```bash
uv run jupyter notebook notebooks/notebook.ipynb
```

Run cells from top to bottom. The first cell locates the project root, so the
notebook works from either the repository root or `notebooks/`.

> Some stages process millions of rows. A full run can take substantial time and
> memory. Generated datasets and metrics are cached under `data/processed/`.

## Sections

| Section | Contents |
| --- | --- |
| 1. Project objective | Two-stage framing: demand recovery then forecasting. |
| 2. Environment and setup | Project-root check and dependency instructions. |
| 3. Data inspection, cleaning and EDA | Raw-data validation, hourly demand, matched stockouts, cleaning summary. |
| 4. Historical demand recovery | Artificial-mask dataset, feature groups, Ridge experiments, tuning, hybrid strategy and held-out recovery test. |
| 5. Forecasting dataset | Lag/rolling/stockout/horizon features and embargoed splits. |
| 6. Naive baseline | Validation metrics for the rule-based baseline. |
| 7. Linear forecasting model | Ridge training, alpha tuning and feature-importance analysis. |
| 8. Tree-based forecasting models | Decision tree, random forest and XGBoost tuning on the validation partition. |
| 9. Final evaluation | Naive vs Ridge vs Random Forest on the untouched test partition. |
| 10. Reproducibility | Generated artifact locations and entrypoint notes. |

## Relationship to the scripts

The notebook calls the same modules used from the command line; it does not
duplicate their implementations. Markdown cells describe each module's inputs,
processing, outputs and purpose before the corresponding cell runs.
