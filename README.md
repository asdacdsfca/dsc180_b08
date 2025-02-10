# Comparing Chronos-Bolt and ARIMA Models for Portfolio Optimization in the S\&P 500 Information Technology Sector
This repository contains code for comparing ARIMA and Chronos-Bolt models in forecasting stock returns for the S&P 500 Information Technology sector. It includes implementations for ARIMA, Chronos-Bolt, and sentiment analysis, integrating technical indicators and investor sentiment into a portfolio optimization framework using the Markowitz mean-variance approach.

## Overview
The data source of this repository is from `yfinance`, a historical financial data API from Yahoo Finance.

The data source of our sentiment analysis is in `4reddit_posts.csv`, a historical financial related posts dataset we scraped from Reddit.

## Dependencies

All required dependencies are provided in a ```requrements.txt``` for convenience

(Note that some are custom Python Libraries on Github and therefore are not in ```requrements.txt```. These are provided below:)

```bash
# Install the Chronos library
!pip install git+https://github.com/amazon-science/chronos-forecasting.git
```

## Installation
To run the scripts, you will need to set up the following environment:

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
After setting up the environment, you can run each scripts individually. They are structured so that you can explore each aspect of the time series forecasting pipeline step by step.

1. ARIMA

2. Navigate to the directory where chronos.py is saved and run:
```bash
python Chronos-Bolt.py
```
**Troubleshooting ChronosPipeline**

If you encounter an error related to ```ChronosPipeline```, verify that the model is accessible by running the following:

```python
from chronos import ChronosPipeline
pipeline = ChronosPipeline.from_pretrained('amazon/chronos-t5-small')
```
If you get an error that ChronosPipeline is not found, try installing Chronos explicitly (if available):
```bash
pip install git+https://github.com/amazon-science/chronos-forecasting.git autogluon pandas numpy torch matplotlib yfinance
```

Sample output of ```Chronos```'s prediction of daily price for S&P500 IT Sector from 10/29/2024 to 12/31/2024 is displayed in ```capstone_Chronos.ipynb```.


