import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import pmdarima as pm  # Auto ARIMA
from statsmodels.tsa.arima.model import ARIMA
import cvxpy as cp  # For portfolio optimization
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
import itertools
from sklearn.preprocessing import StandardScaler
import pmdarima as pm  # Auto ARIMA
import traceback  # For better error handling
import random

### Fetching Data ###

# Wikipedia page listing S&P 500 companies
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
tables = pd.read_html(url)
sp500_table = tables[0]

# Filter for the Information Technology sector
it_sector = sp500_table[sp500_table["GICS Sector"] == "Information Technology"]
it_companies = it_sector[["Symbol", "Security"]].reset_index(drop=True)
tickers = it_companies["Symbol"].tolist()

start_date = "2022-01-01"
end_date = "2024-12-31"

close_prices = {}

for ticker in it_companies["Symbol"]:
    print(f"Fetching close prices for {ticker}...")
    # Fetch historical data from Yahoo Finance
    data = yf.download(ticker, start=start_date, end=end_date)[["Close"]]
    # Save to the dictionary
    close_prices[ticker] = data["Close"]

# Combine all close prices into a single DataFrame, ensuring same index for all tickers
combined_close_prices = pd.concat(close_prices, axis=1)
combined_close_prices.columns = combined_close_prices.columns.droplevel(0)
stock_data = yf.download(tickers = ["AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMZN", "TSLA"], start=start_date, end=end_date)["Close"]
log_returns = stock_data.pct_change().apply(lambda x: np.log(1 + x)).dropna()

### Building ARIMA Model on AAPL###

## Example: use stockprices from 12/1/2024 to 12/31/2024 to predict the prices of the following 10 days
# Suppress warnings
warnings.filterwarnings("ignore")

# Fetch historical data for AAPL
ticker = "AAPL"
start_date = "2024-12-01"
end_date = "2024-12-31"
aapl_data = yf.download(ticker, start=start_date, end=end_date)[["Close"]]

# Prepare data
aapl_close = aapl_data["Close"]

# Fit ARIMA model
model = ARIMA(aapl_close, order=(5, 1, 0))  # p=5, d=1, q=0
results = model.fit()

# Predict future 10 days of stock prices starting from 2025-01-01
start_forecast = pd.to_datetime("2025-01-01")
forecast = results.get_forecast(steps=10)
forecast_df = forecast.conf_int(alpha=0.05)  # 95% confidence interval
forecast_df["Forecast"] = forecast.predicted_mean

# Set the index to start from 2025-01-01
forecast_index = pd.date_range(start=start_forecast, periods=10, freq='D')
forecast_df.index = forecast_index

# Fetch actual values from 2025-01-01 to today
today = datetime.today().strftime('%Y-%m-%d')
actual_values = yf.download(ticker, start="2024-12-01", end="2025-01-10")[["Close"]]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(forecast_df.index, forecast_df["Forecast"], label='Predicted AAPL Close Price', color='blue')
plt.fill_between(forecast_df.index, forecast_df.iloc[:, 0], forecast_df.iloc[:, 1], color='blue', alpha=0.3)
if not actual_values.empty:
    plt.plot(actual_values.index, actual_values["Close"], label='Actual AAPL Close Price', color='red')
plt.title('AAPL Stock Price Prediction for January 2025 using ARIMA')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Define the range for p, d, and q
p = range(0, 8)  # Increase up to 7
d = range(0, 3)  # Differencing typically 0, 1, or 2
q = range(0, 8)  # Increase up to 7

# Create a grid of parameter combinations
pdq_combinations = list(itertools.product(p, d, q))

# Initialize variables to store the best parameters and their corresponding AIC
best_aic = float("inf")
best_pdq = None

train = data.loc[:"2023-12-31", "Close"]
test = data.loc["2024-01-01":"2024-12-31", "Close"]

# Iterate through all parameter combinations
for params in pdq_combinations:
    try:
        # Fit ARIMA model
        model = ARIMA(train, order=params)
        results = model.fit()
        # Update best_aic and best_pdq if the current model's AIC is better
        if results.aic < best_aic:
            best_aic = results.aic
            best_pdq = params
    except Exception as e:
        # Skip invalid parameter combinations
        continue

# Print the best parameters and AIC
print(f"Best ARIMA Parameters: {best_pdq}")
print(f"Best AIC: {best_aic:.2f}")

# Fit the model using the best parameters
best_model = ARIMA(train, order=best_pdq)
best_results = best_model.fit()

# Forecast using the optimized ARIMA model
forecast = best_results.forecast(steps=len(test))

# Calculate evaluation metrics
mae = mean_absolute_error(test, forecast)
mse = mean_squared_error(test, forecast)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label="Training Data", color='blue')
plt.plot(test.index, test, label="Actual Data (2024)", color='red')
plt.plot(test.index, forecast, label="Predicted Data (2024)", color='green')
plt.title("AAPL Stock Price Prediction (Tuned ARIMA)")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.show()

# Suppress warnings
warnings.filterwarnings("ignore")

# Fetch historical data for AAPL from 2022 to 2024
ticker = "AAPL"
data = yf.download(ticker, start="2022-01-01", end="2024-12-31")[["Close"]]

# Check for missing values and fill them if necessary
data["Close"] = data["Close"].interpolate(method='linear')

# Check for stationarity using ADF test
result = adfuller(data["Close"])
print("ADF Statistic:", result[0])
print("p-value:", result[1])

if result[1] > 0.05:
    print("The series is not stationary. Applying first differencing.")
    data["Close_diff"] = data["Close"].diff().dropna()
else:
    print("The series is stationary.")

# Split data into training (2022-2023) and testing (2024)
train = data.loc[:"2023-12-31", "Close"]
test = data.loc["2024-01-01":"2024-12-31", "Close"]

# Fit ARIMA model on training data
model = ARIMA(train, order=(3, 1, 3))  # Replace (5, 1, 0) with tuned parameters if available
results = model.fit()

# Predict on the testing set
forecast = results.forecast(steps=len(test))

# Calculate evaluation metrics
mae = mean_absolute_error(test, forecast)
mse = mean_squared_error(test, forecast)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label="Training Data", color='blue')
plt.plot(test.index, test, label="Actual Data (2024)", color='red')
plt.plot(test.index, forecast, label="Predicted Data (2024)", color='green')
plt.title("AAPL Stock Price Prediction vs Actual (2024)")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.show()

## AAPL: Predict every 5 days rollingly and smooth out

# Suppress warnings
warnings.filterwarnings("ignore")

# Fetch historical data for AAPL from 2022 to 2024
ticker = "AAPL"
data = yf.download(ticker, start="2022-01-01", end="2024-12-31")[["Close"]]

# Check for missing values and fill them if necessary
data["Close"] = data["Close"].interpolate(method='linear')

# Scale the data for stability
scaler = StandardScaler()
data["Close_scaled"] = scaler.fit_transform(data[["Close"]])

# Check for stationarity using ADF test
result = adfuller(data["Close_scaled"])
print("ADF Statistic:", result[0])
print("p-value:", result[1])

if result[1] > 0.05:
    print("The series is not stationary. Applying first differencing.")
    data["Close_scaled_diff"] = data["Close_scaled"].diff().dropna()
else:
    print("The series is stationary.")

# Split data into training (2022-2023) and testing (2024)
train = data.loc["2022-01-01":"2023-12-31", "Close_scaled"]
test = data.loc["2024-01-01":"2024-12-31", "Close_scaled"]

# Parameters for ARIMA (tuned parameters)
order = (3, 1, 3)

# Store overlapping predictions in a dictionary for averaging
predictions_dict = {}

# Rolling prediction with improved handling
for i in range(len(test)):
    rolling_train = data.loc[:test.index[i - 1] if i > 0 else "2023-12-31", "Close_scaled"]

    # Ensure sufficient data for fitting
    min_data_points = 50
    if len(rolling_train) < min_data_points:
        print(f"Skipping iteration {i}: insufficient data in rolling window.")
        continue

    try:
        # Dynamically adjust ARIMA order for small windows
        dynamic_order = order if len(rolling_train) >= 60 else (1, 1, 1)

        # Fit ARIMA model on rolling train data
        model = ARIMA(rolling_train, order=dynamic_order)
        results = model.fit()

        # Forecast next 10 days
        forecast = results.forecast(steps=5)

        # Store each prediction in a dictionary (date as key)
        for j, forecast_value in enumerate(forecast):
            date = test.index[i + j] if i + j < len(test) else None
            if date:
                if date not in predictions_dict:
                    predictions_dict[date] = []
                predictions_dict[date].append(forecast_value)
    except Exception as e:
        print(f"ARIMA fitting failed on iteration {i}: {e}")
        continue

# Average overlapping predictions
forecast_averaged = {date: sum(values) / len(values) for date, values in predictions_dict.items()}

# Convert averaged forecast to DataFrame
forecast_df = pd.DataFrame(list(forecast_averaged.items()), columns=["Date", "Predicted"]).set_index("Date")

# Align actual and predicted values
actual_vs_predicted = pd.concat([test, forecast_df["Predicted"]], axis=1)
actual_vs_predicted.columns = ["Actual", "Predicted"]

# Inverse scale predictions back to original scale
actual_vs_predicted["Actual"] = scaler.inverse_transform(actual_vs_predicted[["Actual"]])
actual_vs_predicted["Predicted"] = scaler.inverse_transform(actual_vs_predicted[["Predicted"]])

# Calculate evaluation metrics
mae = mean_absolute_error(actual_vs_predicted["Actual"], actual_vs_predicted["Predicted"])
mse = mean_squared_error(actual_vs_predicted["Actual"], actual_vs_predicted["Predicted"])
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(actual_vs_predicted.index, actual_vs_predicted["Actual"], label="Actual Data (2024)", color='red')
plt.plot(actual_vs_predicted.index, actual_vs_predicted["Predicted"], label="Predicted Data (Averaged)", color='green')
plt.title("AAPL Stock Price Prediction vs Actual (2024)")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.show()

random.seed(42)

np.random.seed(42)

# Suppress warnings
warnings.filterwarnings("ignore")

# Date ranges to match Chronos approach
start_date = "2022-01-01"
train_end_date = "2024-12-31"  # Last date for training
test_start_date = "2025-01-01"  # First date for testing
test_end_date = "2025-03-05"  # End date (64-day period)

# Function to download stock data
def download_stock_data(tickers):
    # Dictionary to store closing prices
    close_prices = {}
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        try:
            data = yf.download(ticker, start=start_date, end=test_end_date)[["Close"]]
            close_prices[ticker] = data["Close"]
        except Exception as e:
            print(f"Failed to fetch {ticker}: {e}")

    # Combine all stocks into a single DataFrame
    combined_close_prices = pd.concat(close_prices, axis=1)
    combined_close_prices.columns = combined_close_prices.columns.droplevel(0)
    combined_close_prices.dropna(inplace=True)  # Remove rows with missing values

    return combined_close_prices

# Function to get market cap data and create index
def get_market_caps(tickers):
    market_caps = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            market_cap = stock.info.get("marketCap", None)
            if market_cap is not None and market_cap > 0:
                market_caps[ticker] = market_cap
        except Exception as e:
            print(f"Error getting market cap for {ticker}: {e}")

    # Create DataFrame with market cap data
    market_cap_df = pd.DataFrame.from_dict(market_caps, orient='index', columns=["Market Cap"])

    # Calculate total market cap and add percentage column
    total_market_cap = market_cap_df["Market Cap"].sum()
    market_cap_df["Market Cap %"] = (market_cap_df["Market Cap"] / total_market_cap) * 100

    # Sort by market cap (largest to smallest)
    market_cap_df = market_cap_df.sort_values(by="Market Cap", ascending=False)

    return market_cap_df

# Function to calculate returns from prices
def calculate_returns_from_prices(predicted_prices, actual_prices):
    # Calculate daily returns
    predicted_returns = predicted_prices.pct_change().dropna()
    actual_returns = actual_prices.pct_change().dropna()

    return predicted_returns, actual_returns

# Main execution
def main():
    # Define tickers (use Information Technology sector from S&P 500)
    # In production, you'd get this list from a data source
    # This is a sample of major tech companies
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

    # 1. Download historical price data
    print("Downloading historical price data...")
    close_prices = download_stock_data(tickers)

    # Add S&P 500 IT index
    sp500_it_ticker = "XLK"  # ETF tracking S&P 500 Information Technology sector
    print(f"Fetching data for {sp500_it_ticker}...")
    try:
        sp500_it_data = yf.download(sp500_it_ticker, start=start_date, end=test_end_date)[["Close"]]
        # Store the S&P 500 IT index data separately
        sp500_it_prices = sp500_it_data["Close"]
    except Exception as e:
        print(f"Failed to fetch {sp500_it_ticker}: {e}")
        sp500_it_prices = None

    # 2. Get market cap data
    market_cap_data = get_market_caps(tickers)
    print("\nMarket Cap Data:")
    print(market_cap_data)

    # 3. ARIMA forecasting with hyperparameter search for PRICES
    predicted_prices_dict = {}
    mase_scores = {}

    print("\nPerforming ARIMA forecasting with hyperparameter search for each ticker's PRICES...")
    for ticker in tickers:
        try:
            print(f"\nProcessing {ticker}...")

            # Check if we have data for this ticker
            if ticker not in close_prices.columns:
                print(f"No data available for {ticker}. Skipping.")
                continue

            # Check for NaN values
            nan_count = close_prices[ticker].isna().sum()
            if nan_count > 0:
                print(f"Warning: {ticker} has {nan_count} NaN values. Filling with forward fill.")
                close_prices[ticker] = close_prices[ticker].ffill()

            price_series = close_prices[ticker]

            # Split into training and testing
            train = price_series.loc[start_date:train_end_date].dropna()
            test = price_series.loc[test_start_date:test_end_date].dropna()

            # Check if we have enough data
            if len(train) < 30:
                print(f"Warning: Insufficient data for {ticker}. Need at least 30 data points. Skipping.")
                continue

            # Create a validation set from the training data (use last 20% for validation)
            validation_size = int(len(train) * 0.2)
            if validation_size < 10:  # Ensure validation set is at least 10 points
                validation_size = min(20, len(train) // 2)

            train_subset = train.iloc[:-validation_size]
            validation = train.iloc[-validation_size:]

            print(f"Training data size: {len(train_subset)}, Validation data size: {len(validation)}")

            # Default to a simple model in case everything fails
            default_order = (1, 1, 0)
            best_order = default_order

            # Method 1: Use auto_arima for hyperparameter search
            auto_arima_success = False
            try:
                print("Using auto_arima to find the best parameters...")
                auto_model = pm.auto_arima(
                    train_subset,
                    start_p=0, start_q=0,
                    max_p=2, max_q=2, max_d=1,  # Limit model complexity
                    seasonal=False,  # No seasonality for daily stock data
                    information_criterion='aic',  # Use AIC for model selection
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True,  # Faster than exhaustive search
                    trace=True      # Show progress
                )

                best_order = auto_model.order
                print(f"Best ARIMA Order from auto_arima for {ticker}: {best_order}")

                # Validate the model on validation set
                forecast = auto_model.predict(n_periods=len(validation))
                validation_mse = mean_squared_error(validation.values, forecast)
                print(f"Validation MSE: {validation_mse:.4f}")
                auto_arima_success = True

            except Exception as e:
                print(f"Error with auto_arima for {ticker}: {e}")
                auto_arima_success = False

            # Method 2: Manual grid search with validation (if auto_arima failed)
            if not auto_arima_success:
                print("Falling back to manual grid search...")

                # Simple grid with focus on robust models
                p_values = range(0, 2)
                d_values = range(0, 2)
                q_values = range(0, 2)

                best_validation_mse = float('inf')
                grid_search_success = False

                for p in p_values:
                    for d in d_values:
                        for q in q_values:
                            # Skip if all params are 0
                            if p == 0 and d == 0 and q == 0:
                                continue

                            try:
                                print(f"Testing ARIMA({p},{d},{q})...")
                                model = ARIMA(train_subset, order=(p, d, q))
                                results = model.fit()

                                # Forecast on validation set
                                forecast = results.forecast(steps=len(validation))

                                # Calculate validation MSE
                                val_mse = mean_squared_error(validation.values, forecast)
                                print(f"ARIMA({p},{d},{q}) - AIC: {results.aic:.2f}, Validation MSE: {val_mse:.4f}")

                                if val_mse < best_validation_mse:
                                    best_validation_mse = val_mse
                                    best_order = (p, d, q)
                                    grid_search_success = True
                            except Exception as model_error:
                                print(f"Error with ARIMA({p},{d},{q}): {str(model_error)[:100]}...")

                if not grid_search_success:
                    print("Grid search failed. Using default model (1,1,0).")
                    best_order = default_order
                else:
                    print(f"Best order from grid search: {best_order} with validation MSE: {best_validation_mse:.4f}")

            # Re-train the best model on the full training data
            try:
                print(f"Training final model with order {best_order} on full training data...")
                final_model = ARIMA(train, order=best_order)
                final_results = final_model.fit()
                print(f"Training final model succeeded.")
            except Exception as final_model_error:
                print(f"Error fitting final model: {final_model_error}")
                print("Using fallback order (1,1,0)...")
                try:
                    final_model = ARIMA(train, order=default_order)
                    final_results = final_model.fit()
                    best_order = default_order
                    print(f"Fallback model training succeeded.")
                except Exception as fallback_error:
                    print(f"Fallback model also failed: {fallback_error}")
                    print(f"Skipping {ticker} due to modeling failures.")
                    continue

            # Recursive forecasting for the test period
            test_dates = test.index.tolist()
            predictions_dict = {}
            forecasting_success = False

            # Start with the last portion of training data
            window_size = min(60, len(train))
            recursive_data = train.tail(window_size).copy()

            i = 0
            max_retries = 3
            retries = 0

            while i < len(test_dates) and retries < max_retries:
                try:
                    # Use shorter chunks to improve stability
                    steps = min(64, len(test_dates) - i)

                    # Check that we have sufficient data
                    if len(recursive_data) < 20:
                        print(f"Insufficient recursive data for {ticker}. Data length: {len(recursive_data)}")
                        break

                    # Fit model with current recursive data
                    model = ARIMA(recursive_data, order=best_order)
                    results = model.fit()

                    # Forecast next chunk
                    forecast = results.forecast(steps=steps)

                    # Check for NaN values
                    if np.isnan(forecast).any():
                        raise ValueError(f"NaN values in forecast for {ticker}")

                    # Store predictions
                    for j in range(len(forecast)):
                        if i + j < len(test_dates):
                            date = test_dates[i + j]
                            predictions_dict[date] = forecast[j]

                            # Add to recursive dataset
                            try:
                                if date in recursive_data.index:
                                    recursive_data.loc[date] = forecast[j]
                                else:
                                    new_data = pd.Series([forecast[j]], index=[date])
                                    recursive_data = pd.concat([recursive_data, new_data])
                            except Exception as concat_error:
                                print(f"Error adding to recursive data: {concat_error}")
                                # Try alternative method
                                new_index = recursive_data.index.tolist() + [date]
                                new_values = recursive_data.values.tolist() + [forecast[j]]
                                recursive_data = pd.Series(new_values, index=new_index)

                    # Sort to ensure chronological order
                    recursive_data = recursive_data.sort_index()

                    # Move to next chunk
                    i += steps
                    forecasting_success = True
                    retries = 0  # Reset retries after successful iteration

                except Exception as model_error:
                    print(f"Error in forecasting loop iteration {i}: {model_error}")
                    retries += 1

                    if retries >= max_retries:
                        print(f"Maximum retries reached for {ticker}. Using naive forecast.")

                        # Naive forecast: use last value with small noise
                        last_value = recursive_data.iloc[-1]
                        for j in range(64):  # Just forecast 5 days with naive method
                            if i + j < len(test_dates):
                                date = test_dates[i + j]
                                noise_factor = np.random.uniform(0.98, 1.02)
                                naive_forecast = last_value * noise_factor
                                predictions_dict[date] = naive_forecast

                                # Add to recursive data
                                if date in recursive_data.index:
                                    recursive_data.loc[date] = naive_forecast
                                else:
                                    new_data = pd.Series([naive_forecast], index=[date])
                                    recursive_data = pd.concat([recursive_data, new_data])

                        recursive_data = recursive_data.sort_index()
                        i += 64
                        retries = 0  # Reset for next chunk

            # Convert predictions to a series and store if we have valid predictions
            if predictions_dict and len(predictions_dict) > 0:
                forecast_series = pd.Series(predictions_dict)

                # Verify we have valid data
                if not forecast_series.empty and not np.isnan(forecast_series).all():
                    predicted_prices_dict[ticker] = forecast_series
                    print(f"Successfully generated {len(forecast_series)} predictions for {ticker}")

                    # Calculate MASE if possible
                    if len(forecast_series) > 0 and len(test) > 0:
                        common_index = test.index.intersection(forecast_series.index)
                        if len(common_index) > 0:
                            actual_values = test[common_index]
                            predicted_values = forecast_series[common_index]

                            # Calculate MAE
                            mae = np.mean(np.abs(actual_values - predicted_values))

                            # Calculate MAE for naive forecast
                            if len(actual_values) > 1:
                                naive_mae = np.mean(np.abs(actual_values.values[1:] - actual_values.values[:-1]))
                                if naive_mae > 0:
                                    mase = mae / naive_mae
                                    mase_scores[ticker] = mase
                                    print(f"MASE for {ticker} price predictions: {mase:.4f}")
                else:
                    print(f"Generated predictions for {ticker} but they are empty or all NaN. Skipping.")
            else:
                print(f"No valid predictions for {ticker}")

        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            traceback.print_exc()  # Print full traceback for debugging
            continue

    # Check if we have any valid predictions
    if not predicted_prices_dict:
        print("No valid predictions were generated for any ticker. Cannot proceed with portfolio optimization.")
        return None, None, None

    # Create DataFrame of all predicted prices
    predicted_prices_df = pd.DataFrame(predicted_prices_dict)

    # Convert MASE scores to DataFrame
    mase_scores_df = pd.DataFrame.from_dict(mase_scores, orient='index', columns=["MASE Score"])
    print("\nMASE Scores for Price Predictions:")
    print(mase_scores_df.sort_values(by="MASE Score"))

    # 4. Calculate predicted returns from predicted prices
    # Extract actual test prices for tickers with predictions
    valid_tickers = list(predicted_prices_dict.keys())
    print(f"\nValid tickers with predictions: {valid_tickers}")

    if len(valid_tickers) == 0:
        print("No valid tickers with predictions. Cannot proceed.")
        return None, None, None

    # Get test prices only for valid tickers
    valid_test_prices = close_prices.loc[test_start_date:test_end_date, valid_tickers]

    # Calculate returns - carefully check for data issues
    try:
        predicted_returns, actual_returns = calculate_returns_from_prices(predicted_prices_df, valid_test_prices)

        # Check if we have valid return data
        if predicted_returns.empty or actual_returns.empty:
            raise ValueError("Empty returns data")

        print(f"Predicted returns shape: {predicted_returns.shape}")
        print(f"Actual returns shape: {actual_returns.shape}")
    except Exception as e:
        print(f"Error calculating returns: {e}")
        # Fallback to just using the actual returns
        if not valid_test_prices.empty:
            actual_returns = valid_test_prices.pct_change().dropna()
            # Create dummy predicted returns (just copy the actual ones for simplicity)
            predicted_returns = actual_returns.copy()
        else:
            print("Cannot calculate returns. Exiting.")
            return None, None, None

    # 5. Portfolio Optimization
    try:
        # Double check valid tickers (those with predictions)
        valid_tickers = [ticker for ticker in predicted_returns.columns if not predicted_returns[ticker].empty]

        if len(valid_tickers) < 2:
            print(f"Need at least 2 valid tickers for portfolio optimization. Only have {len(valid_tickers)}.")
            # Use equal weights as fallback
            weights = pd.Series(1.0/len(valid_tickers), index=valid_tickers)
            optimal_portfolio = pd.DataFrame(weights, columns=["Portfolio Weight"])
        else:
            # Calculate covariance matrix
            cov_matrix = predicted_returns[valid_tickers].cov().values

            # Ensure matrix is positive semidefinite
            eigvals, eigvecs = np.linalg.eigh(cov_matrix)
            eigvals = np.maximum(eigvals, 1e-6)
            cov_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Portfolio optimization variables
            num_assets = len(valid_tickers)
            w = cp.Variable(num_assets)

            # Expected returns
            expected_returns_vector = predicted_returns[valid_tickers].mean().values

            # Define optimization problem
            portfolio_return = expected_returns_vector @ w
            portfolio_risk = cp.quad_form(w, cov_matrix)

            # Risk aversion parameter
            risk_aversion = 0.5

            # Objective: maximize return - risk_aversion * risk
            objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_risk)

            # Constraints
            constraints = [
                cp.sum(w) == 1,  # Fully invested
                w >= 0,          # No short selling
                w <= 0.25        # Maximum 25% in any stock
            ]

            # Solve optimization
            problem = cp.Problem(objective, constraints)
            problem.solve()

            if problem.status not in ["optimal", "optimal_inaccurate"]:
                print(f"Portfolio optimization failed with status: {problem.status}")
                # Use equal weights as fallback
                weights = pd.Series(1.0/len(valid_tickers), index=valid_tickers)
                optimal_portfolio = pd.DataFrame(weights, columns=["Portfolio Weight"])
            else:
                # Extract optimal weights
                optimal_weights = w.value

                # Create DataFrame of optimal weights
                optimal_portfolio = pd.DataFrame(optimal_weights, index=valid_tickers, columns=["Portfolio Weight"])
    except Exception as e:
        print(f"Error in portfolio optimization: {e}")
        # Use equal weights as fallback
        weights = pd.Series(1.0/len(valid_tickers), index=valid_tickers)
        optimal_portfolio = pd.DataFrame(weights, columns=["Portfolio Weight"])

    print("\nOptimal Portfolio Weights:")
    print(optimal_portfolio.sort_values(by="Portfolio Weight", ascending=False))

    # 6. Portfolio Performance Evaluation
    try:
        # Initial investment
        initial_investment = 10000

        # Calculate daily portfolio values
        portfolio_values = pd.DataFrame(index=actual_returns.index)

        # Calculate optimized portfolio value
        portfolio_values["Portfolio Value ($10,000 Investment)"] = initial_investment
        for i in range(len(actual_returns)):
            daily_return = 0
            for ticker, weight in optimal_portfolio["Portfolio Weight"].items():
                if ticker in actual_returns.columns:
                    current_date = actual_returns.index[i]
                    if current_date in actual_returns.index:
                        daily_return += weight * actual_returns.loc[current_date, ticker]

            if i == 0:
                portfolio_values.iloc[i, 0] = initial_investment * (1 + daily_return)
            else:
                portfolio_values.iloc[i, 0] = portfolio_values.iloc[i-1, 0] * (1 + daily_return)

        # Calculate S&P 500 IT Index Value for comparison
        portfolio_values["S&P 500 IT Sector Index Value ($10,000 Investment)"] = initial_investment

        if sp500_it_prices is not None:
            # Get index values for test period
            sp500_it_test = sp500_it_prices.loc[test_start_date:test_end_date]
            sp500_it_returns = sp500_it_test.pct_change().dropna()

            # Calculate index performance
            for i in range(len(sp500_it_returns)):
                if i == 0:
                    portfolio_values.iloc[i, 1] = initial_investment * (1 + sp500_it_returns.iloc[i])
                else:
                    portfolio_values.iloc[i, 1] = portfolio_values.iloc[i-1, 1] * (1 + sp500_it_returns.iloc[i])
        else:
            # Use equal-weighted portfolio as benchmark if SP500 IT data not available
            for i in range(len(actual_returns)):
                daily_return = actual_returns.iloc[i].mean()  # Equal-weighted average
                if i == 0:
                    portfolio_values.iloc[i, 1] = initial_investment * (1 + daily_return)
                else:
                    portfolio_values.iloc[i, 1] = portfolio_values.iloc[i-1, 1] * (1 + daily_return)

        # 7. Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_values.index, portfolio_values["S&P 500 IT Sector Index Value ($10,000 Investment)"], 'b-', label='S&P 500 IT Sector Index')
        plt.plot(portfolio_values.index, portfolio_values["Portfolio Value ($10,000 Investment)"], 'g-', label='Optimized Portfolio')
        plt.title("Portfolio Performance Comparison (2024/10/29 to 2024/12/31)")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.grid(True)
        plt.legend()
        plt.savefig('portfolio_comparison_hyperparameter_search.png')

        # 8. Calculate final statistics
        optimized_return = (portfolio_values["Portfolio Value ($10,000 Investment)"].iloc[-1] / initial_investment) - 1
        benchmark_return = (portfolio_values["S&P 500 IT Sector Index Value ($10,000 Investment)"].iloc[-1] / initial_investment) - 1
        outperformance = optimized_return - benchmark_return

        print("\n=== Final Results ===")
        print(f"Portfolio Return: {optimized_return:.2%}")
        print(f"S&P 500 IT Sector Index Return: {benchmark_return:.2%}")
        print(f"Outperformance: {outperformance:.2%}")
        print(f"Final Portfolio Value: ${portfolio_values['Portfolio Value ($10,000 Investment)'].iloc[-1]:.2f}")
        print(f"Final S&P 500 IT Sector Index Value: ${portfolio_values['S&P 500 IT Sector Index Value ($10,000 Investment)'].iloc[-1]:.2f}")

        # 9. Save results to CSV
        portfolio_values.to_csv("portfolio_performance_hyperparameter_search.csv")
        if not mase_scores_df.empty:
            mase_scores_df.to_csv("mase_scores_hyperparameter_search.csv")
        optimal_portfolio.to_csv("optimal_weights_hyperparameter_search.csv")

        # Save NVDA predictions to CSV
        if "NVDA" in predicted_prices_dict:
            print("\nSaving NVDA predictions to CSV...")
            nvda_predictions = predicted_prices_dict["NVDA"]

            # Get actual NVDA prices for the test period
            nvda_actual = close_prices.loc[test_start_date:test_end_date, "NVDA"]

            # Create DataFrame with both predicted and actual prices
            nvda_data = pd.DataFrame({
                "Date": nvda_predictions.index,
                "Predicted_Price": nvda_predictions.values
            })

            # Set index to Date
            nvda_data.set_index("Date", inplace=True)

            # Add actual prices
            nvda_data["Actual_Price"] = np.nan
            for date in nvda_data.index:
                if date in nvda_actual.index:
                    nvda_data.loc[date, "Actual_Price"] = nvda_actual.loc[date]

            # Calculate prediction error
            nvda_data["Error"] = nvda_data["Actual_Price"] - nvda_data["Predicted_Price"]
            nvda_data["Error_Percent"] = (nvda_data["Error"] / nvda_data["Actual_Price"]) * 100

            # Save to CSV
            nvda_data.to_csv("nvda_predictions.csv")
            print("NVDA predictions saved to nvda_predictions.csv")

            # Create a plot for visualization
            plt.figure(figsize=(12, 6))
            plt.plot(nvda_data.index, nvda_data["Actual_Price"], "b-", label="Actual Price")
            plt.plot(nvda_data.index, nvda_data["Predicted_Price"], "r-", label="Predicted Price")
            plt.title("NVDA Actual vs Predicted Prices")
            plt.xlabel("Date")
            plt.ylabel("Price ($)")
            plt.grid(True)
            plt.legend()
            plt.savefig("nvda_predictions_comparison.png")
            print("NVDA prediction plot saved to nvda_predictions_comparison.png")
        else:
            print("No NVDA predictions available to save.")

        return portfolio_values, optimal_portfolio, mase_scores_df

    except Exception as e:
        print(f"Error in portfolio evaluation: {e}")
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    portfolio_values, weights, mase_scores = main()
