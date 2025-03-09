# Comparing Chronos-Bolt and ARIMA Models for Portfolio Optimization in the S\&P 500 Information Technology Sector
This repository contains code for comparing ARIMA and Chronos-Bolt models in forecasting stock returns for the S&P 500 Information Technology sector. It includes implementations for ARIMA, Chronos-Bolt, and sentiment analysis, integrating technical indicators and investor sentiment into a portfolio optimization framework using the Markowitz mean-variance approach.

## Our Deliverables
You can find our website [here](https://asdacdsfca.github.io/dsc180-b08-website/)

You can find our poster [here](https://drive.google.com/file/d/1V6RnXS4tDHc7dhsLZYl8quiOU0kCbego/view?usp=sharing)
## Overview
The data source of this repository is from `yfinance`, a historical financial data API from Yahoo Finance.

The data source of our sentiment analysis is in `4reddit_posts.csv`, a historical financial related posts dataset we scraped from Reddit.

## Dependencies

All required dependencies are provided in a ```requrements.txt``` for convenience

(Note that some are custom Python Libraries on Github and therefore are not in ```requrements.txt```. These are provided below:)

```bash
# Install the Chronos library
pip install git+https://github.com/amazon-science/chronos-forecasting.git
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
```bash
python ARIMA.py
```
2. Navigate to the directory where chronos.py is saved and run:
```bash
pip install git+https://github.com/amazon-science/chronos-forecasting.git
```
**Note that below steps require you to have sentiment data ready. If you do not want to customize the sentiment analysis settings or scrape new/specific data, you can access our provided sample data by:*

```bash
cd sample sentiment dataset
```

**To Get the Predictions from Chronos, run:**

```bash
python Chronos_prediction.py
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

Sample output of ```Chronos```'s prediction of daily price for S&P500 IT Sector from 01/01/2025 to 03/05/2025 is displayed in ```capstone_Chronos.ipynb```.

### If you are interested using the scrapper, please follow the below steps:

🚀 Description of the scrapper: It scrapes Reddit posts from financial subreddits, analyzes them for relevant stock market discussions, and filters them using OpenAI's GPT-4o-mini model.

1. Set Up API Keys:
Create a `.env` file in the root directory and add your API keys:
```bash
# Reddit API Keys
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=your_reddit_user_agent

# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key
```
2. Run the automated pipeline by:
```bash
./process_reddit.sh
```
