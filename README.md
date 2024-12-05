# Stock Price Prediction in the Electric Vehicle Sector: Advanced Forecasting Methodologies and Market Analysis
In the rapidly evolving landscape of financial technology, this repository represents a cutting-edge exploration of advanced machine learning techniques for stock market analysis and investment strategy.

## Overview
Each notebook in this repository covers a different method of stock price prediction.
The data source of this repository is from `yfinance`, a historical financial data API from Yahoo Finance.

- `time_series_chronos.ipynb`: Investigates the use of the Chronos time series forecasting model for predicting stock prices in the electric vehicle sector.

- `time_series_TTM.ipynb`: Explores the application of Tiny Time Mixers (TTM), an innovative time series forecasting approach, to predict volatile stock movements.

- `tech_indicators.ipynb`: Delves into the implementation and analysis of technical indicators such as RSI, MACD, and EMA to capture market trends and generate investment signals.

- `fund_metrics.ipynb`: Evaluates fundamental financial metrics, including earnings per share and price-to-earnings ratios, to assess company valuation and growth potential.

- `evasive_detection.ipynb`: Presents a novel sentiment analysis approach that identifies linguistic patterns of evasiveness in corporate communications, providing insights into potential market risks.

- `Portfolio_Optimization.ipynb`: Combines advanced portfolio optimization techniques, including Mean-Variance and Conditional Value-at-Risk, to develop adaptive investment strategies that balance risk and reward.

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

