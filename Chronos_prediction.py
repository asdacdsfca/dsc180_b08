import pandas as pd
import numpy as np
import yfinance as yf
from chronos import ChronosPipeline
import matplotlib.pyplot as plt
import torch
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from sklearn.metrics import mean_squared_error

# **Tickers are defined here** **so that each company is processed.**
tickers = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "NVDA",  # Nvidia
    "AVGO",  # Broadcom
    "ORCL",  # Oracle
    "CRM",   # Salesforce
    "CSCO",  # Cisco
    "IBM",   # IBM
    "ADBE",  # Adobe
    "QCOM",  # Qualcomm
    "AMD",   # Advanced Micro Devices
    "PLTR"   # Palantir
]

def get_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    if df.empty:
        print(f"Warning: No data found for {ticker}. It may be delisted or unavailable.")
        return None
    df.index = df.index.tz_localize(None)  # **Ensure timezone consistency** **for proper date handling.**
    return df['Close']

def calculate_mase(training_data, forecast_actual, forecast_predicted):
    if forecast_actual is None or forecast_predicted is None:
        return np.nan
    mae = np.mean(np.abs(forecast_actual - forecast_predicted))
    last_value = training_data.iloc[-1] if isinstance(training_data, pd.Series) else training_data[-1]
    naive_errors = np.abs(forecast_actual - last_value)
    naive_mae = np.mean(naive_errors)
    return mae / naive_mae

def calculate_wql(actual, forecast_array):
    if actual is None or forecast_array is None:
        return np.nan
    quantiles = np.linspace(0.1, 0.9, 9)
    wql_scores = []
    forecast_array_trimmed = forecast_array[:, :len(actual)]
    for q in quantiles:
        quantile_forecast = np.percentile(forecast_array_trimmed, q * 100, axis=0)
        error = np.maximum(q * (actual - quantile_forecast), (q - 1) * (actual - quantile_forecast))
        wql_scores.append(np.mean(error))
    return np.mean(wql_scores)

# **Global dictionaries store forecast outputs** **and performance metrics for later analysis.**
predicted_prices = {}
expected_returns = {}
two_month_returns = {}

start_date = "2022-01-01"
end_date = "2024-12-31"
forecast_start = "2025-01-01"
forecast_end = "2025-03-05"
forecast_dates = pd.date_range(start=forecast_start, end=forecast_end)
prediction_length = len(forecast_dates)


# **Generate a naive forecast** **by repeating the last observed value over the forecast horizon.**
def forecast_naive(training_data, steps, forecast_index):
    last_value = training_data.iloc[-1]
    return pd.Series([last_value] * steps, index=forecast_index)

def process_company(ticker):
    print(f"\nProcessing {ticker}...")
    start_date_local = "2022-01-01"
    end_date_local = "2024-12-31"
    forecast_start_local = "2025-01-01"
    forecast_end_local = "2025-03-05"

    all_stock = get_stock_data(ticker, start_date_local, forecast_end_local)
    training_data = get_stock_data(ticker, start_date_local, end_date_local)

    if training_data is None or all_stock is None:
        return {
            'Ticker': ticker,
            'MASE': np.nan,
            'WQL': np.nan,
            'Expected Daily Return': np.nan,
            'Predicted Return': np.nan
        }

    forecast_dates_local = pd.date_range(start=forecast_start_local, end=forecast_end_local)
    steps = len(forecast_dates_local)

    # **Obtain Naive forecast** **by using the last observed value.**
    naive_forecast = forecast_naive(training_data, steps, forecast_dates_local)

    # **Generate Chronos forecast using the pretrained Chronos model.**
    stock_pipeline = ChronosPipeline.from_pretrained('amazon/chronos-t5-small',
                                                      device_map="cpu",
                                                      torch_dtype=torch.bfloat16)
    context = torch.tensor(training_data.values).unsqueeze(0)
    stock_forecast = stock_pipeline.predict(context=context, prediction_length=steps, num_samples=36)
    forecast_array = stock_forecast.numpy().squeeze()
    chronos_median = np.median(forecast_array, axis=0)
    lower_bound = np.percentile(forecast_array, 10, axis=0)
    upper_bound = np.percentile(forecast_array, 90, axis=0)

    chronos_forecast_df = pd.DataFrame({
        'Chronos Forecast': chronos_median,
        'Lower Bound': lower_bound,
        'Upper Bound': upper_bound
    }, index=forecast_dates_local)

    # **Plot historical data and forecasts** **together for comparison.**
    plt.figure(figsize=(12, 6))
    plt.plot(all_stock.index, all_stock.values, label='Historical Data', color='blue', linewidth=1)
    plt.plot(naive_forecast.index, naive_forecast.values, label='Naive Forecast',
             color='purple', linewidth=2, linestyle='--')
    plt.plot(chronos_forecast_df.index, chronos_forecast_df['Chronos Forecast'],
             label='Chronos Forecast', color='red', linewidth=2)
    plt.fill_between(chronos_forecast_df.index, chronos_forecast_df['Lower Bound'],
                     chronos_forecast_df['Upper Bound'], color='red', alpha=0.2, label='80% Chronos PI')
    plt.xlim([pd.Timestamp(forecast_start_local) - pd.Timedelta(days=30),
              pd.Timestamp(forecast_end_local) + pd.Timedelta(days=10)])
    # **Limit y-axis** **to range 100 to 160.**
    plt.ylim([100, 160])
    plt.title(f'{ticker} Stock Price Forecast Comparison')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

    actual_forecast = all_stock[forecast_start_local:forecast_end_local]

    # **Calculate performance metrics** **using the Chronos forecast for consistency.**
    mase = calculate_mase(training_data, actual_forecast, chronos_forecast_df['Chronos Forecast'])
    wql = calculate_wql(actual_forecast.values, forecast_array)

    predicted_prices[ticker] = chronos_forecast_df['Chronos Forecast'].values
    forecast_series = pd.Series(predicted_prices[ticker], index=forecast_dates_local)
    expected_daily_return = forecast_series.mean()
    two_month_cumulative_return = forecast_series.sum()

    expected_returns[ticker] = expected_daily_return
    two_month_returns[ticker] = two_month_cumulative_return

    print(f"MASE: {mase:.4f}, WQL: {wql:.4f}, Expected Daily Return: {expected_daily_return:.4f}, Predicted Return: {two_month_cumulative_return:.4f}")

    return {
        'Ticker': ticker,
        'MASE': mase,
        'WQL': wql,
        'Expected Daily Return': expected_daily_return,
        'Predicted Return': two_month_cumulative_return
    }

if __name__ == "__main__":
    # Process all tickers
    results = [process_company(ticker) for ticker in tickers]
    summary_results2 = pd.DataFrame(results)

    # Convert predicted price results into DataFrames
    predicted_prices_df2 = pd.DataFrame(predicted_prices, index=forecast_dates)
    expected_returns_df2 = pd.DataFrame.from_dict(expected_returns, orient='index', columns=["Expected Daily Return"])
    two_month_returns_df2 = pd.DataFrame.from_dict(two_month_returns, orient='index', columns=["Predicted Return"])

    # Display final results
    print(summary_results2)

