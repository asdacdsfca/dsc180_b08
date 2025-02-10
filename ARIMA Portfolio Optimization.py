!pip install git+https://github.com/amazon-science/chronos-forecasting.git
import pandas as pd
import numpy as np
import yfinance as yf
from chronos import ChronosPipeline
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import torch
import requests
import pmdarima as pm  # Auto ARIMA
from statsmodels.tsa.arima.model import ARIMA
import cvxpy as cp  # For portfolio optimization
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller

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

### Building ARIMA Model ###

## Example: use stockprices from 12/1/2024 to 12/31/2024 to predict the prices of the following 10 days
import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import warnings
from datetime import datetime

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

## Hyperparameter Tuning for AAPL
import itertools
from statsmodels.tsa.arima.model import ARIMA

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

## Use Best Parameters From Tuning on AAPL
import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller
import warnings

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
import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
import warnings

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

## Now Do the Same Things on Other Stocks

# Suppress warnings
warnings.filterwarnings("ignore")

# Load log_returns dataset (Assuming you have it as a Pandas DataFrame)
log_returns = log_returns  # Replace with your dataset

# Define tickers (Modify as needed)
tickers = log_returns.columns.tolist()

# Define split dates
train_end_date = "2024-06-30"  # First 2.5 years for training
test_start_date = "2024-07-01"  # Last 6 months for testing
test_end_date = "2024-12-31"

# Storage for expected returns and 6-month cumulative returns
expected_returns = {}
six_month_returns = {}

# Rolling ARIMA forecasting for each ticker
for ticker in tickers:
    try:
        print(f"Processing {ticker}...")

        # Check stationarity using ADF test
        result = adfuller(log_returns[ticker])
        if result[1] > 0.05:
            print(f"{ticker} is not stationary. Applying first differencing.")
            log_returns[ticker] = log_returns[ticker].diff().dropna()

        # Split into training (First 2.5 years) and testing (Last 6 months)
        train = log_returns.loc["2022-01-01":train_end_date, ticker].dropna()
        test = log_returns.loc[test_start_date:test_end_date, ticker].dropna()

        # **Find Best ARIMA Order Using Auto-ARIMA**
        auto_arima_model = pm.auto_arima(train, seasonal=False, stepwise=True,
                                 suppress_warnings=True, trace=True)
        best_order = auto_arima_model.order  # Extract (p, d, q)

        print(f"Best ARIMA Order for {ticker}: {best_order}")

        # Store rolling predictions
        predictions_dict = {}

        for i in range(len(test)):
            rolling_train = log_returns.loc[:test.index[i - 1] if i > 0 else train_end_date, ticker]

            # Ensure enough data for fitting
            min_data_points = 50
            if len(rolling_train) < min_data_points:
                print(f"Skipping {ticker} at iteration {i}: insufficient data.")
                continue

            try:
                # Fit ARIMA model with best (p, d, q)
                model = ARIMA(rolling_train, order=best_order)
                results = model.fit()

                # Forecast next 5 days
                forecast = results.forecast(steps=5)

                # Store each prediction in a dictionary (date as key)
                for j, forecast_value in enumerate(forecast):
                    date = test.index[i + j] if i + j < len(test) else None
                    if date:
                        if date not in predictions_dict:
                            predictions_dict[date] = []
                        predictions_dict[date].append(forecast_value)

            except Exception as e:
                print(f"ARIMA fitting failed for {ticker} at iteration {i}: {e}")
                continue

        # Average overlapping predictions
        forecast_averaged = {date: sum(values) / len(values) for date, values in predictions_dict.items()}

        # Compute final expected return for the ticker (Mean of all forecasts)
        expected_returns[ticker] = np.mean(list(forecast_averaged.values()))

        # Compute 6-month cumulative return (Sum of predicted log returns)
        six_month_returns[ticker] = np.sum(list(forecast_averaged.values()))

    except Exception as e:
        print(f"Skipping {ticker} due to error: {e}")
        continue

# Convert expected returns and 6-month returns to DataFrames
expected_returns_df = pd.DataFrame.from_dict(expected_returns, orient='index', columns=["Expected Daily Return"])
six_month_returns_df = pd.DataFrame.from_dict(six_month_returns, orient='index', columns=["6-Month Cumulative Return"])

# Convert log returns to actual returns
# six_month_returns_df["Actual Return"] = np.exp(six_month_returns_df["6-Month Cumulative Return"]) - 1
six_month_returns_df["Actual Return"] = np.exp(six_month_returns_df["6-Month Cumulative Return"]) - 1

### Portfolio Optimization ###

# Compute covariance matrix only for selected tickers
cov_matrix = log_returns[tickers].cov().values  # Convert to NumPy array
print("Covariance Matrix:\n", cov_matrix)
# Ensure portfolio weights variable
num_assets = len(tickers)
w = cp.Variable((num_assets, 1))

# Define optimization problem
portfolio_return = expected_returns_df.T.values @ w
portfolio_risk = cp.quad_form(w, cov_matrix)

# Risk aversion parameter (adjustable)
risk_aversion = 0.5  # Adjust for more or less risk

# Objective: maximize return while penalizing risk
objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_risk)

# Constraints
constraints = [
    cp.sum(w) == 1,  # Fully invested portfolio
    w >= 0           # No short-selling constraint
]

# Solve optimization problem
problem = cp.Problem(objective, constraints)
problem.solve()

# Extract optimal weights
optimal_weights = w.value
optimal_portfolio = pd.DataFrame(optimal_weights, index=tickers, columns=["Portfolio Weight"])

# === Compute Investment Growth Over 6 Months ===
total_investment = 10000  # $10,000 total investment

# Merge portfolio weights with returns
investment_df = optimal_portfolio.merge(six_month_returns_df, left_index=True, right_index=True)

# Compute initial investment per stock
investment_df["Investment Amount"] = total_investment * investment_df["Portfolio Weight"]

# Compute final value per stock
investment_df["Final Value"] = investment_df["Investment Amount"] * (1 + investment_df["Actual Return"])

# Compute total final portfolio value
total_final_value = investment_df["Final Value"].sum()

# Display Results
from IPython.display import display
print("\nInvestment Breakdown Over 6 Months:")
display(investment_df)

print(f"\nTotal Portfolio Value After 6 Months: ${total_final_value:,.2f}")

# Plot Portfolio Growth
plt.figure(figsize=(10, 5))
plt.bar(investment_df.index, investment_df["Final Value"], color='skyblue', label="Final Value")
plt.bar(investment_df.index, investment_df["Investment Amount"], color='gray', alpha=0.5, label="Initial Investment")
plt.title("Portfolio Growth Over 6 Months")
plt.ylabel("Value ($)")
plt.legend()
plt.xticks(rotation=45)
plt.show()


