# Perishable Demand Forecasting

An intelligent demand estimation and forecasting platform designed to help retailers understand customer demand, anticipate stockouts, optimize inventory and reduce perishable product waste.

The system estimates **underlying product demand** and forecasts future demand, accounting for sales that may be hidden when products are temporarily unavailable.

## Project Structure

```text
train.py                        # train the final model -> models/
predict.py                      # start the forecast web service
src/perishable_demand_forecasting/
├── config.py                   # centralised project paths and constants
├── data/                       # dataset download and daily dataset preparation
├── analysis/                   # dataset inspection and stockout analysis
├── recovery/                   # latent demand recovery: datasets, features, models, evaluation
├── forecasting/                # demand forecasting models and evaluation
└── api/                        # online inference: model loading and Flask routes
```

Run workflow modules from the repository root, e.g.:

```bash
uv run python -m perishable_demand_forecasting.analysis.inspect_raw_dataset
uv run python train.py
uv run python predict.py
```

### Run with Docker

Train the model first so `models/forecasting/final_demand_forecast.joblib` exists, then build and run the service:

```bash
docker build -t perishable-demand-forecasting .
docker run --rm -p 8000:8000 perishable-demand-forecasting
```

The service is available at `http://localhost:8000` (`GET /health`, `POST /predict`).

## The Problem

Managing inventory for perishable products is a difficult balancing act.

If a retailer orders too little, products may sell out before customers have finished buying them. This results in lost sales, reduced product availability and dissatisfied customers.

If a retailer orders too much, products may remain unsold until they approach their expiry or shelf-life limit, resulting in markdowns, waste and lost margins.

The problem is made harder by the data retailers typically use to make these decisions:

> **Historical sales are not always a true representation of customer demand.**

Consider a product that sells 50 units before a store runs out of stock. The retailer's system records 50 sales. However, 70 customers may have wanted to purchase the product.

The remaining 20 units of potential demand are invisible in the sales data.

A forecasting system that learns directly from historical sales can therefore learn the wrong lesson:

```text
Low recorded sales
        ↓
Low demand forecast
        ↓
Insufficient inventory
        ↓
Stockout
        ↓
Even more lost sales
```

Perishable products make this problem particularly important because retailers must simultaneously manage **product availability, inventory costs and product waste**.

## The Solution

Perishable Demand Forecasting uses machine learning to estimate underlying demand and forecast future demand for individual products and stores.

Rather than treating every historical sales observation as an accurate representation of demand, the system considers the conditions under which those sales occurred.

It combines information such as:

- Historical sales
- Product availability
- Stockout events
- Promotions and discounts
- Day of week
- Holidays
- Weather
- Store characteristics
- Product characteristics

The resulting forecasts aim to answer a more useful business question:

> **How much of this product are customers likely to demand during the forecast period?**

Rather than simply:

> **How much did the store sell last time?**

## Product Workflow

```text
                  Retail Data
                       |
                       ▼
             ┌─────────────────────┐
             │  Data Preparation   │
             │  & Validation       │
             └──────────┬──────────┘
                        |
                        ▼
             ┌─────────────────────┐
             │  Stockout Detection │
             └──────────┬──────────┘
                        |
                        ▼
             ┌─────────────────────┐
             │  Demand Estimation  │
             │  & Recovery         │
             └──────────┬──────────┘
                        |
                        ▼
             ┌─────────────────────┐
             │  Demand Forecasting │
             └──────────┬──────────┘
                        |
                        ▼
             ┌─────────────────────┐
             │  Forecast API       │
             └──────────┬──────────┘
                        |
                        ▼
              Inventory Planning
               & Decision Support
```

The forecasting service provides expected demand for a specific product, store and future time period.

For example:

```text
Store:       Store 042
Product:     Fresh Product A
Forecast:    72 units
Period:      Tomorrow
Confidence:  68–78 units
```

Forecasts can become inputs into inventory replenishment, purchasing and planning workflows.

## Dataset

The initial data source is **FreshRetailNet-50K**, a large-scale retail dataset released by Dingdong Inc. for research into demand estimation and forecasting in fresh retail.

[FreshRetailNet-50K on Hugging Face](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K)

### Dataset Characteristics

| Attribute            | Description                                   |
| -------------------- | --------------------------------------------- |
| Store-product series | 50,000                                        |
| Stores               | 898                                           |
| Cities               | 18                                            |
| Perishable SKUs      | 865                                           |
| Training period      | 90 days                                       |
| Evaluation period    | 7 days                                        |
| Sales                | Daily and hourly sales information            |
| Stock availability   | Hourly stock-status information               |
| Promotions           | Discount and activity information             |
| Calendar             | Holiday information                           |
| Weather              | Temperature, precipitation, humidity and wind |

The dataset contains naturally occurring stockout observations, making it valuable for developing demand estimation and forecasting methods that account for constrained sales.

### Research Reference

FreshRetailNet-50K was introduced in:

**FreshRetailNet-50K: A Stockout-Annotated Censored Demand Dataset for Latent Demand Recovery and Forecasting in Fresh Retail**

[Research paper](https://arxiv.org/abs/2505.16319)

## Machine Learning Approach

The system treats demand forecasting as more than a straightforward time-series prediction problem.

### 1. Detect Constrained Sales

Identify periods where observed sales may have been affected by product unavailability.

### 2. Estimate Underlying Demand

Estimate the demand that could have occurred if the product had remained available.

### 3. Forecast Future Demand

Use recovered demand together with temporal, product, store, promotional and environmental features to forecast future demand.

### 4. Evaluate Business-Relevant Performance

Evaluate forecasting accuracy across multiple time horizons, products, stores and stockout conditions.

The modelling approach is designed to evolve as additional data, features and forecasting techniques become available.

## Business Value

More accurate demand estimates and forecasts can help retailers make better inventory decisions.

### Reduce Stockouts

Better estimates of demand help identify products that are likely to sell out, supporting more effective replenishment planning.

### Reduce Waste

Improved forecasts help retailers avoid unnecessary inventory of products with short shelf lives.

### Improve Product Availability

Maintaining appropriate inventory levels helps ensure products are available when customers want them.

### Improve Inventory Planning

Forecasts provide a consistent, data-driven input for replenishment, purchasing and inventory management workflows.

Ultimately, the goal is to help retailers find a better balance between:

**Availability ↔ Inventory Cost ↔ Product Waste**

## Forecasting Capabilities

The system is designed around store-product-level forecasting and supports the development of capabilities such as:

- Product demand forecasts
- Store-level demand forecasts
- Multiple forecast horizons
- Demand trends and seasonality analysis
- Stockout risk identification
- Forecast confidence intervals
- Historical versus forecast demand analysis
- Forecast performance monitoring
- Inventory planning and replenishment support

## Model Evaluation

Forecasting performance is evaluated using chronological training, validation and test datasets.

Evaluation metrics include:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Forecast bias

Performance is also analyzed across:

- Forecast horizons
- Products
- Stores
- Demand levels
- Stockout conditions
- Time periods

### Current Forecasting Results

The current forecasting implementation evaluates two approaches:

- **Naive Forecasting:** Uses the most recent recovered daily demand as the forecast.
- **Ridge Regression:** Uses historical demand and contextual features to predict future demand.

Final evaluation on the held-out test dataset:

| Metric |     Naive | Ridge Regression |
| ------ | --------: | ---------------: |
| MAE    |  0.512812 |     **0.407412** |
| RMSE   |  0.814507 |     **0.648432** |
| Bias   | -0.037006 |        -0.114638 |

Ridge Regression achieved:

- **20.55% lower MAE**
- **20.39% lower RMSE**

These results demonstrate the value of incorporating historical demand patterns and contextual information into the forecasting process.

The evaluation also tracks negative predictions and forecast bias to identify areas requiring further improvement.

## Production Architecture

The system is designed to evolve into a production-ready forecasting platform.

```text
                  ┌──────────────────┐
                  │   Retail Data    │
                  └────────┬─────────┘
                           |
                           ▼
                  ┌──────────────────┐
                  │  Data Pipeline   │
                  └────────┬─────────┘
                           |
                           ▼
                  ┌──────────────────┐
                  │ Feature Store /  │
                  │ Feature Pipeline │
                  └────────┬─────────┘
                           |
                           ▼
                  ┌──────────────────┐
                  │  Forecast Model  │
                  └────────┬─────────┘
                           |
                           ▼
                  ┌──────────────────┐
                  │  Forecast API    │
                  └────────┬─────────┘
                           |
                 ┌─────────┴─────────┐
                 ▼                   ▼
          Planning Tools       Retail Systems
```

The architecture separates data processing, feature engineering, model development and forecast delivery.

This allows forecasting models to be replaced, improved or extended without requiring major changes to the surrounding application.

## Project Goals

Perishable Demand Forecasting aims to provide a reliable demand estimation and forecasting capability that can:

- Estimate underlying demand despite historical stockouts.
- Forecast future demand at store-product level.
- Incorporate relevant contextual information.
- Improve inventory planning and replenishment decisions.
- Provide measurable forecasting accuracy.
- Expose forecasts through a production-ready API.
- Support model monitoring and continuous improvement.
- Scale to additional retailers and datasets.

## Data License

FreshRetailNet-50K is released under the **CC BY 4.0** license.

Users should review the dataset's licensing terms before using the data commercially.

---

**Perishable Demand Forecasting — Turning retail sales data into better demand intelligence.**
