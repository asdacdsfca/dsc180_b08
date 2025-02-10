import pandas as pd
import numpy as np
import yfinance as yf
import torch
import matplotlib.pyplot as plt
from chronos import ChronosPipeline

def get_sp500_it_companies():
    """Fetches the list of S&P 500 companies and filters for the Information Technology sector."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    sp500_table = tables[0]
    it_sector = sp500_table[sp500_table["GICS Sector"] == "Information Technology"]
    return it_sector[["Symbol", "Security"]].reset_index(drop=True)

def get_stock_data(ticker, start_date, end_date):
    """Fetches historical stock data for a given ticker."""
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    if df.empty:
        print(f"Warning: No data found for {ticker}. It may be delisted or unavailable.")
        return None
    df.index = df.index.tz_localize(None)  # Ensure timezone consistency
    return df['Close']

def calculate_mase(actual, predicted):
    """Computes the Mean Absolute Scaled Error (MASE)."""
    if actual is None or predicted is None:
        return np.nan
    mae = np.mean(np.abs(actual - predicted))
    naive_mae = np.mean(np.abs(actual.values[1:] - actual.values[:-1]))
    return mae / naive_mae

def calculate_wql(actual, forecast_array):
    """Computes the Weighted Quantile Loss (WQL)."""
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

def process_company(company_name, ticker):
    """Processes a company's stock data, runs a Chronos model, and evaluates performance."""
    print(f"\nProcessing {company_name} ({ticker}):")
    start_date, end_date, forecast_start, forecast_end = "2022-01-01", "2024-10-02", "2024-10-29", "2024-12-31"
    
    all_stock = get_stock_data(ticker, start_date, forecast_end)
    stock = get_stock_data(ticker, start_date, end_date)
    
    if stock is None or all_stock is None:
        return {'Company': company_name, 'Ticker': ticker, 'MASE': np.nan, 'WQL': np.nan}
    
    stock_pipeline = ChronosPipeline.from_pretrained('amazon/chronos-t5-small', device_map="cpu", torch_dtype=torch.bfloat16)
    context = torch.tensor(stock.values).unsqueeze(0)
    stock_forecast = stock_pipeline.predict(context=context, prediction_length=64, num_samples=100)
    
    forecast_dates = pd.date_range(start=forecast_start, end=forecast_end)
    forecast_array = stock_forecast.numpy().squeeze()
    
    median_forecast = np.median(forecast_array, axis=0)
    lower_bound, upper_bound = np.percentile(forecast_array, 10, axis=0), np.percentile(forecast_array, 90, axis=0)
    
    stock_forecast_df = pd.DataFrame({'Median Forecast': median_forecast, 'Lower Bound': lower_bound, 'Upper Bound': upper_bound}, index=forecast_dates)
    
    plt.figure(figsize=(12, 6))
    plt.plot(all_stock.index, all_stock.values, label='Historical Data', color='blue')
    plt.plot(stock_forecast_df.index, stock_forecast_df['Median Forecast'], label='Median Forecast', color='red')
    plt.fill_between(stock_forecast_df.index, stock_forecast_df['Lower Bound'], stock_forecast_df['Upper Bound'], color='red', alpha=0.2, label='80% Prediction Interval')
    plt.title(f'{company_name} ({ticker}) Stock Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    actual_forecast = all_stock[forecast_start:forecast_end]
    mase, wql = calculate_mase(actual_forecast, stock_forecast_df['Median Forecast']), calculate_wql(actual_forecast.values, forecast_array)
    
    print(f"MASE: {mase:.4f}, WQL: {wql:.4f}")
    return {'Company': company_name, 'Ticker': ticker, 'MASE': mase, 'WQL': wql}

def main():
    """Main execution function."""
    it_companies = get_sp500_it_companies()
    summary_results = pd.DataFrame([process_company(row["Security"], row["Symbol"]) for _, row in it_companies.iterrows()])
    print(summary_results)
    # print(process_company("Apple", "AAPL"))

if __name__ == "__main__":
    main()
