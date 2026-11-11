# Data

## Source

The project uses **FreshRetailNet-50K**, a large-scale fresh-retail dataset
released by Dingdong Inc. for research into demand estimation and forecasting.

- [Dataset on Hugging Face](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K)
- [Research paper](https://arxiv.org/abs/2505.16319)

Download it with:

```bash
uv run python -m perishable_demand_forecasting.data.download_raw_data
```

## Dataset characteristics

| Attribute | Description |
| --- | --- |
| Store-product series | 50,000 |
| Stores | 898 |
| Cities | 18 |
| Perishable SKUs | 865 |
| Training period | 90 days |
| Evaluation period | 7 days |
| Sales | Daily and hourly sales information |
| Stock availability | Hourly stock-status information |
| Promotions | Discount and activity information |
| Calendar | Holiday information |
| Weather | Temperature, precipitation, humidity and wind |

## Files

| Path | Contents |
| --- | --- |
| `data/raw/freshretailnet/data/train.parquet` | Main modelling period (2024-03-28 to 2024-06-25). |
| `data/raw/freshretailnet/data/eval.parquet` | Evaluation period (2024-06-26 to 2024-07-02). |

The raw data is expanded into hourly arrays covering the 06:00–21:00 operating
window. Daily sales must reconcile with the hourly sales arrays; the inspection
step checks schema, coverage, nulls, duplicate keys and value ranges.

## Why stockouts matter

A product that sells out records fewer sales than customers demanded. Roughly
44% of training rows are stockout-affected and about 4% are fully out of stock,
so treating observed sales as demand would bias the forecast downward.

The dataset does not include verified lost-demand ground truth. The recovery
stage therefore uses artificial masking: fully in-stock observations are treated
as known demand, and stockout masks borrowed from other observations are applied
to their hourly sales. This provides a controlled reconstruction target.

## Splits

| Split | Purpose |
| --- | --- |
| Train | Fit models. |
| Validation | Feature-group selection, alpha tuning and model selection. |
| Test | Final held-out evaluation only. |

Recovery and forecasting splits are chronological, and forecasting partitions
include a seven-day embargo to reduce temporal leakage.

## License

FreshRetailNet-50K is released under the **CC BY 4.0** license. Review the
dataset's terms before commercial use.
