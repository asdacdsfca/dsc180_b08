# Stock Price Prediction in the Electric Vehicle Sector: Forecasting Methodologies and Market Analysis
In the rapidly evolving landscape of financial technology, this repository represents a cutting-edge exploration of advanced machine learning techniques for stock market analysis and investment strategy.

## Overview
Each notebook in this repository explores a different method of stock price prediction.
The data source of this repository is from `yfinance`, a historical financial data API from Yahoo Finance.

## Dependencies

All required dependencies are provided in cells throughout the notebook. You may install these as you run through the notebook. We have also included a ```requrements.txt``` for convenience

(Note that some are custom Python Libraries and therefore are not in ```requrements.txt```)

## Installation
To run the notebooks, you will need to set up the following environment:

1. Clone the repository:
```bash
git clone https://github.com/asdacdsfca/dsc180_b08.git
cd dsc180_b08
```

2. Install the packages:
```bash
pip install -r requirements.txt
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

