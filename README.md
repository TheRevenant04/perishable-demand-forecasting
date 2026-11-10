# Perishable Demand Forecasting

An intelligent demand forecasting system for perishable retail products.

The system helps retailers estimate **true product demand and forecast future demand**, including demand that may be hidden when products are temporarily out of stock.

---

## The Problem

Managing inventory for perishable products is a difficult balancing act.

If a retailer orders too little, products sell out before customers have finished buying them. This results in lost sales and poor product availability.

If a retailer orders too much, products may remain unsold until they approach their expiry or shelf-life limit, resulting in markdowns, waste, and lost margin.

The problem is made harder by the data retailers typically use to make these decisions:

> **Historical sales are not always a true representation of customer demand.**

Consider a product that sells 50 units before a store runs out of stock. The retailer's system records 50 sales. However, 70 customers may have wanted to purchase the product.

The remaining 20 units of potential demand are invisible in the sales data.

This means a forecasting system that learns directly from historical sales can learn the wrong lesson:

```text
Low recorded sales
        ↓
Low forecast
        ↓
Low inventory
        ↓
Stockout
        ↓
Even more lost sales
```

Perishable products make this problem particularly important because retailers must simultaneously manage **availability, inventory cost, and product waste**.

---

## The Solution

The system uses machine learning to estimate **underlying demand** and forecast future demand for individual products and stores.

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

The resulting forecast is intended to answer a more useful business question:

> **How much of this product are customers likely to demand during the forecast period?**

rather than simply:

> **How much did the store sell last time?**

---

## Product Workflow

```text
              Retail Data
                   │
                   ▼
        ┌─────────────────────┐
        │ Data Preparation    │
        │ & Validation        │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ Stockout Detection  │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ Demand Estimation   │
        │ & Recovery          │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ Demand Forecasting  │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ Forecast API        │
        └──────────┬──────────┘
                   │
                   ▼
        Inventory Planning
        & Decision Support
```

The forecasting service can provide expected demand for a specific **product, store, and future time period**.

For example:

```text
Store:       Store 042
Product:     Fresh Product A
Forecast:    72 units
Period:      Tomorrow
Confidence:  68–78 units
```

The forecast can then become an input into inventory replenishment and planning workflows.

---

## Dataset

The initial data source is **FreshRetailNet-50K**, a large-scale retail dataset released by Dingdong Inc. for research into demand estimation and forecasting in fresh retail.

[FreshRetailNet-50K on Hugging Face](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K?utm_source=chatgpt.com)

The dataset contains information covering:

| Attribute            | Description                                   |
| -------------------- | --------------------------------------------- |
| Store-product series | 50,000                                        |
| Stores               | 898                                           |
| Cities               | 18                                            |
| Perishable SKUs      | 865                                           |
| Time span            | 90 days                                       |
| Sales                | Hourly sales information                      |
| Stock availability   | Hourly stock-status information               |
| Promotions           | Discount and activity information             |
| Calendar             | Holiday information                           |
| Weather              | Temperature, precipitation, humidity and wind |

The dataset contains naturally occurring stockout observations, making it particularly valuable for developing demand forecasts that account for constrained sales.

### Research Reference

FreshRetailNet-50K was introduced in:

**FreshRetailNet-50K: A Stockout-Annotated Censored Demand Dataset for Latent Demand Recovery and Forecasting in Fresh Retail**

[Research paper](https://arxiv.org/abs/2505.16319?utm_source=chatgpt.com)

---

## Machine Learning Approach

The system treats demand forecasting as more than a straightforward time-series prediction problem.

### 1. Detect constrained sales

Identify periods where observed sales may have been affected by product unavailability.

### 2. Estimate underlying demand

Estimate the demand that could have occurred if the product had remained available.

### 3. Forecast future demand

Use recovered demand together with temporal, product, store, promotional, and environmental features to forecast future demand.

### 4. Evaluate business-relevant performance

The system evaluates not only overall forecasting accuracy, but also how well it performs during and around stockout events.

The exact modelling approach is intentionally not fixed. Different forecasting and demand-recovery methods will be evaluated against strong baselines.

---

## Business Value

A more accurate demand forecast can help retailers make better inventory decisions.

### Reduce stockouts

Better estimates of demand can help identify products that are likely to sell out.

### Reduce waste

Better forecasts can reduce unnecessary inventory of products with short shelf lives.

### Improve availability

Maintaining the right inventory levels helps ensure products are available when customers want them.

### Improve inventory planning

Forecasts can provide a consistent, data-driven input for replenishment and purchasing decisions.

Ultimately, the goal is to help retailers find a better balance between:

**Availability ↔ Inventory Cost ↔ Product Waste**

---

## Forecasting Capabilities

The system is designed around store-product level forecasting and can be extended to support:

- Product demand forecasts
- Store-level demand forecasts
- Multiple forecast horizons
- Demand trends
- Stockout risk identification
- Forecast confidence intervals
- Historical vs. forecast demand analysis
- Forecast performance monitoring

---

## Evaluation

Model performance will be measured using time-series validation and forecasting metrics such as:

- MAE
- RMSE
- sMAPE
- Forecast bias

Performance will also be analyzed across:

- Products
- Stores
- Demand levels
- Stockout conditions
- Time periods

A key performance question is:

> **Does explicitly accounting for stockouts produce better demand forecasts than forecasting directly from observed sales?**

---

## Production Architecture

The system is designed to evolve into a production-ready forecasting service.

```text
                    ┌─────────────────┐
                    │  Retail Data    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Data Pipeline   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Feature Store / │
                    │ Feature Pipeline│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Forecast Model  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Forecast API    │
                    └────────┬────────┘
                             │
                 ┌───────────┴───────────┐
                 ▼                       ▼
          Planning Tools          Retail Systems
```

The architecture allows the forecasting model to be replaced or improved without requiring major changes to the surrounding application.

---

## Technology

The implementation will use a modern machine-learning and production engineering stack.

The technology choices will evolve as the system develops, but the architecture is intended to support:

- Python
- Pandas / Polars
- Scikit-learn and/or specialized forecasting libraries
- REST API
- Containerized deployment
- Automated model training
- Model versioning
- Monitoring and evaluation

---

## Project Goals

The product aims to provide a reliable demand forecasting capability that can:

- Estimate demand despite historical stockouts
- Forecast future demand at store-product level
- Incorporate relevant contextual information
- Provide measurable forecast accuracy
- Expose forecasts through a production-ready API
- Support model monitoring and future retraining
- Scale to additional retailers and datasets

---

## Data License

FreshRetailNet-50K is released under the **CC BY 4.0** license.

Users should review the dataset's licensing terms before using the data commercially.

---
