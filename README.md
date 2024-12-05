# Evaluating Time-Series Forecasting Models for Stock Price Prediction

This repository contains the implementation and analysis of various time series forecasting models, utilizing technical indicators and sentiment analysis. The goal is to explore how can techniques make informed decisions about investments.

## Overview
Each notebook in this repository covers a different component of the time series forecasting pipeline.

The data source of this repository is from ```yfinance``` a historical financial data API from Yahoo Finance.


## Dependencies

The required dependencies are provided in cells throughout the notebook. You may install these as you run through the notebook.


## Installation
To run the notebooks, you will need to set up the following environment:

1. Clone the repository:
```bash
git clone https://github.com/asdacdsfca/dsc180_b08.git
cd dsc180_b08
```

## Usage
After setting up the environment, you can run each notebook individually. They are structured so that you can explore each aspect of the time series forecasting pipeline step by step.

1.  ```Chronos```'s prediction of daily price for EV sector from 9/1/2024 to 9/30/2024
```bash
jupyter notebook time_series_chronos.ipynb
```

2. ```Tiny Time Mixers```'s prediction of daily price for EV sector from 9/1/2024 to 9/30/2024
```bash
jupyter notebook time_series_TTM.ipynb
```

3. Daily technical analysis dashboard for EV sectors from 10/1/2024 to 10/11/2024.
```bash
jupyter notebook technical_indicators.ipynb
```

4. Compare fundamental metrics for ten EV industry tickers
```bash
jupyter notebook fund_metrics.ipynb
```

5. Evasive Detection Prompts Exploration
```bash
jupyter notebook evasive_detection.ipynb
```

6. Portfolio Optimization Technique Exploration
```bash
jupyter notebook Portfolio_Optimization.ipynb
``` 

