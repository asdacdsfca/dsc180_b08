# Comparing Chronos-Bolt and ARIMA Models for Portfolio Optimization in the S\&P 500 Information Technology Sector
This repository contains code for comparing ARIMA and Chronos-Bolt models in forecasting stock returns for the S&P 500 Information Technology sector. It includes implementations for ARIMA, Chronos-Bolt, and sentiment analysis, integrating technical indicators and investor sentiment into a portfolio optimization framework using the Markowitz mean-variance approach.

## Overview
The data source of this repository is from `yfinance`, a historical financial data API from Yahoo Finance.

## Dependencies

All required dependencies are provided in cells throughout the notebook. You may install these as you run through the notebook. We have also included a ```requrements.txt``` for convenience

(Note that some are custom Python Libraries on Github and therefore are not in ```requrements.txt```. These are provided below:)

```bash
# Install the Chronos library
!pip install git+https://github.com/amazon-science/chronos-forecasting.git
```

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

