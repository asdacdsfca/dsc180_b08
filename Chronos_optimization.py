#!/usr/bin/env python3
"""
Portfolio Optimization with Sentiment Analysis

This script implements a portfolio optimization strategy for tech stocks,
incorporating sentiment analysis to adjust expected returns. It compares
the optimized portfolio against a market cap weighted portfolio.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns
import yfinance as yf

# Mapping between stock symbols and company names
symbol_to_name = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
    "AVGO": "Broadcom",
    "ORCL": "Oracle",
    "CRM": "Salesforce",
    "CSCO": "Cisco",
    "IBM": "IBM",
    "ADBE": "Adobe",
    "QCOM": "Qualcomm",
    "AMD": "Advanced Micro Devices",
    "PLTR": "Palantir"
}

# Reverse mapping for lookup by company name
name_to_symbol = {v: k for k, v in symbol_to_name.items()}

# Market cap data for initialization (in billions USD, approximate as of late 2024)
market_caps = {
    "Apple": 2900,
    "Microsoft": 3100,
    "Nvidia": 2200,
    "Broadcom": 750,
    "Oracle": 320,
    "Salesforce": 260,
    "Cisco": 220,
    "IBM": 150,
    "Adobe": 280,
    "Qualcomm": 170,
    "Advanced Micro Devices": 280,
    "Palantir": 70
}

def load_sentiment_data(file1, file2):
    """
    Load and preprocess sentiment analysis data from CSV files.
    
    Args:
        file1: Path to first sentiment data file
        file2: Path to second sentiment data file
        
    Returns:
        DataFrame containing consolidated sentiment data
    """
    sentiment1 = pd.read_csv(file1)
    sentiment2 = pd.read_csv(file2)
    
    # Standardize company names
    sentiment2["company"] = sentiment2["company"].replace("AMD", "Advanced Micro Devices")
    
    # Concatenate data from both files
    sentiment_df = pd.concat([sentiment1, sentiment2], ignore_index=True)
    
    # Filter to keep only companies in our universe
    company_names = list(symbol_to_name.values())
    sentiment_df = sentiment_df[sentiment_df['company'].isin(company_names)]
    
    # Reset index after filtering
    sentiment_df = sentiment_df.reset_index(drop=True)
    
    return sentiment_df

def get_actual_prices(symbols, start_date, end_date):
    """
    Retrieve historical stock prices using yfinance.
    
    Args:
        symbols: List of stock symbols to retrieve
        start_date: Beginning date for data retrieval
        end_date: Ending date for data retrieval
        
    Returns:
        DataFrame with daily prices for all symbols
    """
    # Download data for all symbols at once
    data = yf.download(symbols, start=start_date, end=end_date)
    
    # Extract adjusted close prices
    close_prices = data['Close']
    
    # Reshape the data to long format
    result_df = close_prices.reset_index()
    result_df = result_df.melt(id_vars='Date', var_name='company', value_name='price')
    result_df = result_df.rename(columns={'Date': 'date'})
    
    return result_df

def calculate_returns_and_cov(predicted_prices_df):
    """
    Calculate expected returns and covariance matrix from predicted prices.
    
    Args:
        predicted_prices_df: DataFrame with predicted stock prices
        
    Returns:
        expected_returns: Series of expected daily returns
        cov_matrix: Covariance matrix of returns
        returns: DataFrame of daily returns
    """
    # Pivot the dataframe to have date as index and company as columns
    pivot_df = predicted_prices_df.pivot(index='date', columns='company', values='price')
    
    # Calculate daily returns
    returns = pivot_df.pct_change().dropna()
    
    # Calculate expected returns (mean of daily returns for each stock)
    expected_returns = returns.mean()
    
    # Calculate the covariance matrix of returns
    cov_matrix = returns.cov()
    
    return expected_returns, cov_matrix, returns

def adjust_returns_with_sentiment(expected_returns, sentiment_df, sentiment_weight=0.5):
    """
    Adjust expected returns based on sentiment analysis data.
    
    Args:
        expected_returns: Series of expected returns
        sentiment_df: DataFrame with sentiment analysis data
        sentiment_weight: Weight to apply to sentiment adjustment (0-1)
        
    Returns:
        adjusted_returns: Returns modified by sentiment
        sentiment_adjustment: Dictionary of sentiment adjustment factors
    """
    # If no sentiment data is provided, return unmodified expected returns
    if sentiment_df is None:
        return expected_returns, {company: 1.0 for company in expected_returns.index}

    # Create a sentiment adjustment factor
    sentiment_adjustment = {}
    
    for company in expected_returns.index:
        # Get the sentiment data for this company
        company_sentiment = sentiment_df[sentiment_df['company'] == company]
        
        if company_sentiment.empty:
            sentiment_adjustment[company] = 1.0  # No adjustment if no sentiment data
            continue
        
        # Use the most recent sentiment data
        latest_sentiment = company_sentiment.iloc[-1]
        
        # Calculate adjustment based on bullish percentage
        bullish_pct = latest_sentiment['bullish_pct'] / 100
        
        # Create a sentiment score with more impact
        # Scale from 0.9 to 1.1 for more noticeable effect
        sentiment_score = 1 + (bullish_pct - 0.5) * 0.4
        
        sentiment_adjustment[company] = sentiment_score
    
    # Apply the sentiment adjustment to expected returns with specified weight
    adjusted_returns = expected_returns.copy()
    for company in expected_returns.index:
        if company in sentiment_adjustment:
            # Apply sentiment with the given weight
            adjusted_returns[company] = expected_returns[company] * (1 + (sentiment_adjustment[company] - 1) * sentiment_weight)
    
    print(sentiment_adjustment)
    return adjusted_returns, sentiment_adjustment

def optimize_portfolio(expected_returns, cov_matrix, risk_aversion=2.5, min_weight=0.05):
    """
    Optimize portfolio weights using mean-variance optimization with constraints.
    
    Args:
        expected_returns: Series of expected returns for each asset
        cov_matrix: Covariance matrix of returns
        risk_aversion: Risk aversion parameter (higher = more risk-averse)
        min_weight: Minimum weight for each asset
        
    Returns:
        Series of optimal portfolio weights
    """
    num_assets = len(expected_returns)
    
    # Objective function to minimize: -R + (λ/2)*σ²
    def objective(weights):
        portfolio_return = np.sum(expected_returns * weights)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        return -portfolio_return + (risk_aversion/2) * portfolio_variance
    
    # Constraints: weights sum to 1
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    # Add minimum weight constraints
    for i in range(num_assets):
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[i] - min_weight
        })
    
    # Bounds: each weight must be between min_weight and 1
    bounds = tuple((min_weight, 1) for _ in range(num_assets))
    
    # Initial guess: equal weights
    initial_weights = np.ones(num_assets) / num_assets
    
    # Perform the optimization
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if not result['success']:
        print("Warning: Optimization did not converge. Using equal weights.")
        return pd.Series(initial_weights, index=expected_returns.index)
    
    return pd.Series(result['x'], index=expected_returns.index)

def calculate_portfolio_metrics(weights, expected_returns, cov_matrix):
    """
    Calculate key performance metrics for a portfolio.
    
    Args:
        weights: Series of portfolio weights
        expected_returns: Series of expected returns
        cov_matrix: Covariance matrix of returns
        
    Returns:
        Dictionary containing portfolio metrics
    """
    # Calculate expected portfolio return
    portfolio_return = np.sum(expected_returns * weights)
    
    # Calculate portfolio variance and standard deviation (risk)
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_risk = np.sqrt(portfolio_variance)
    
    # Calculate Sharpe ratio (assuming risk-free rate of 0 for simplicity)
    sharpe_ratio = portfolio_return / portfolio_risk
    
    return {
        'expected_return': portfolio_return,
        'risk': portfolio_risk,
        'sharpe_ratio': sharpe_ratio,
        'annualized_return': (1 + portfolio_return)**252 - 1,  # Annualized (252 trading days)
        'annualized_risk': portfolio_risk * np.sqrt(252)  # Annualized risk
    }

def simulate_portfolio_performance(weights, returns_df):
    """
    Simulate portfolio performance over time given historical returns.
    
    Args:
        weights: Series of portfolio weights
        returns_df: DataFrame with daily returns for each asset
        
    Returns:
        Series with portfolio value over time
    """
    # Calculate portfolio returns for each day
    portfolio_returns = (returns_df * weights).sum(axis=1)
    
    # Calculate cumulative returns (starting with $1)
    portfolio_value = (1 + portfolio_returns).cumprod()
    
    return portfolio_value

def compare_predicted_vs_actual(predicted_prices_df, actual_prices_df, start_date, end_date):
    """
    Compare predicted returns against actual returns for evaluation.
    
    Args:
        predicted_prices_df: DataFrame with predicted prices
        actual_prices_df: DataFrame with actual prices
        start_date: Start date of test period
        end_date: End date of test period
        
    Returns:
        DataFrame comparing predicted and actual returns
    """
    # Filter for test period
    pred_test = predicted_prices_df[(predicted_prices_df['date'] >= start_date) &
                                    (predicted_prices_df['date'] <= end_date)]
    actual_test = actual_prices_df[(actual_prices_df['date'] >= start_date) & 
                                    (actual_prices_df['date'] <= end_date)]
    
    # Get first and last day for each stock
    companies = predicted_prices_df['company'].unique()
    comparison = []
    
    for company in companies:
        # Predicted returns
        pred_company = pred_test[pred_test['company'] == company].sort_values('date')
        if len(pred_company) > 1:
            pred_first = pred_company.iloc[0]['price']
            pred_last = pred_company.iloc[-1]['price']
            pred_return = (pred_last / pred_first) - 1
        else:
            pred_return = 0
        
        # Actual returns
        # Convert company name to stock symbol
        symbol = name_to_symbol.get(company, None)
        
        if symbol is None:
            print(f"Error: No symbol found for company '{company}'")
        else:
            actual_company = actual_test[actual_test['company'] == symbol].sort_values('date')
        
        if len(actual_company) > 1:
            actual_first = actual_company.iloc[0]['price']
            actual_last = actual_company.iloc[-1]['price']
            actual_return = (actual_last / actual_first) - 1
        else:
            actual_return = 0
        
        comparison.append({
            'company': company,
            'predicted_return': pred_return,
            'actual_return': actual_return
        })
    
    return pd.DataFrame(comparison)

def plot_efficient_frontier(expected_returns, cov_matrix, optimal_weights, market_cap_weights=None, n_points=20):
    """
    Plot the efficient frontier, individual assets, and portfolio positions.
    
    Args:
        expected_returns: Series of expected returns
        cov_matrix: Covariance matrix of returns
        optimal_weights: Series of optimal portfolio weights
        market_cap_weights: Series of market cap weights (optional)
        n_points: Number of points to plot on the frontier
        
    Returns:
        Matplotlib figure of the efficient frontier
    """
    # Calculate the efficient frontier
    risk_levels = np.linspace(np.min(np.sqrt(np.diag(cov_matrix)))*0.8, 
                              np.max(np.sqrt(np.diag(cov_matrix)))*1.2, 
                              n_points)
    returns = []
    risks = []
    
    for target_risk in risk_levels:
        # For each risk level, find the portfolio with maximum return
        num_assets = len(expected_returns)
        
        def objective(weights):
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            risk = np.sqrt(portfolio_variance)
            return -np.sum(expected_returns * weights)
        
        # Risk constraint
        risk_constraint = {
            'type': 'eq',
            'fun': lambda x: np.sqrt(np.dot(x.T, np.dot(cov_matrix, x))) - target_risk
        }
        
        # Weights sum to 1 constraint
        sum_constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        
        # Bounds
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Optimize
        initial_weights = np.ones(num_assets) / num_assets
        
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=[risk_constraint, sum_constraint]
        )
        
        if result['success']:
            weights = result['x']
            portfolio_return = np.sum(expected_returns * weights)
            portfolio_risk = target_risk
            
            returns.append(portfolio_return)
            risks.append(portfolio_risk)
    
    # Create efficient frontier plot
    plt.figure(figsize=(12, 8))
    plt.plot(risks, returns, 'b-', linewidth=3, label='Efficient Frontier')
    
    # Plot individual assets
    for i, (asset, ret) in enumerate(expected_returns.items()):
        asset_risk = np.sqrt(cov_matrix.iloc[i, i])
        plt.scatter(asset_risk, ret, marker='o', s=100, label=asset)
    
    # Plot optimized portfolio
    metrics = calculate_portfolio_metrics(optimal_weights, expected_returns, cov_matrix)
    plt.scatter(
        metrics['risk'],
        metrics['expected_return'],
        marker='*',
        s=300,
        color='red',
        label='Optimized Portfolio'
    )
    
    # Plot market cap weighted portfolio if provided
    if market_cap_weights is not None:
        mcap_metrics = calculate_portfolio_metrics(market_cap_weights, expected_returns, cov_matrix)
        plt.scatter(
            mcap_metrics['risk'],
            mcap_metrics['expected_return'],
            marker='s',
            s=200,
            color='green',
            label='Market Cap Portfolio'
        )
    
    plt.title('Efficient Frontier and Assets')
    plt.xlabel('Risk (Standard Deviation)')
    plt.ylabel('Expected Return')
    plt.grid(True)
    plt.legend()
    
    return plt

def optimize_portfolio_with_constraints(predicted_prices_df, sentiment_df, market_caps,
                                        min_weight=0.05, risk_aversion=2.5, sentiment_weight=0.5):
    """
    Run the full portfolio optimization process with sentiment adjustment.
    
    Args:
        predicted_prices_df: DataFrame with predicted prices
        sentiment_df: DataFrame with sentiment data
        market_caps: Dictionary of market capitalizations
        min_weight: Minimum weight for each asset
        risk_aversion: Risk aversion parameter
        sentiment_weight: Weight to apply to sentiment adjustments
        
    Returns:
        Dictionary containing all portfolio optimization results
    """
    # Step 1: Calculate returns and covariance from predicted prices
    expected_returns, cov_matrix, returns_df = calculate_returns_and_cov(predicted_prices_df)
    
    # Step 2: Adjust returns based on sentiment
    adjusted_returns, sentiment_adjustment = adjust_returns_with_sentiment(expected_returns, sentiment_df, sentiment_weight)
    
    # Step 3: Calculate market cap weights for comparison
    total_market_cap = sum(market_caps[company] for company in expected_returns.index)
    market_cap_weights = pd.Series({company: market_caps[company]/total_market_cap for company in expected_returns.index})
    
    # Step 4: Optimize portfolio with adjusted returns and minimum weights
    optimal_weights = optimize_portfolio(
        adjusted_returns,
        cov_matrix,
        risk_aversion=risk_aversion,
        min_weight=min_weight
    )
    
    # Step 5: Calculate portfolio metrics
    optimal_metrics = calculate_portfolio_metrics(optimal_weights, adjusted_returns, cov_matrix)
    mcap_metrics = calculate_portfolio_metrics(market_cap_weights, adjusted_returns, cov_matrix)
    
    # Step 6: Simulate portfolio performance
    optimal_performance = simulate_portfolio_performance(optimal_weights, returns_df)
    mcap_performance = simulate_portfolio_performance(market_cap_weights, returns_df)
    
    # Create comprehensive visualizations
    
    # 1. Efficient Frontier Plot
    frontier_plot = plot_efficient_frontier(adjusted_returns, cov_matrix, optimal_weights, market_cap_weights)
    plt.savefig('efficient_frontier.png')
    plt.close()
    
    # 2. Portfolio Analysis Plot
    plt.figure(figsize=(15, 10))
    
    # Plot weights comparison
    plt.subplot(2, 2, 1)
    comparison_df = pd.DataFrame({
        'Optimal': optimal_weights,
        'Market Cap': market_cap_weights
    })
    comparison_df.plot(kind='bar', ax=plt.gca())
    plt.title('Portfolio Allocation Comparison')
    plt.ylabel('Weight')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    
    # Plot sentiment adjustment factors
    plt.subplot(2, 2, 2)
    sentiment_factors = pd.Series(sentiment_adjustment)
    sentiment_factors.plot(kind='bar', color='purple')
    plt.axhline(y=1, color='r', linestyle='-')
    plt.title('Sentiment Adjustment Factors')
    plt.ylabel('Adjustment Factor')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    
    # Plot performance comparison
    plt.subplot(2, 2, 3)
    performance_df = pd.DataFrame({
        'Optimal Portfolio': optimal_performance,
        'Market Cap Portfolio': mcap_performance
    })
    performance_df.to_csv('performance.csv', index=True)
    performance_df.plot(ax=plt.gca())
    plt.title('Simulated Portfolio Performance')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    
    # Plot expected returns comparison
    plt.subplot(2, 2, 4)
    returns_comparison = pd.DataFrame({
        'Original Returns': expected_returns,
        'Sentiment-Adjusted': adjusted_returns
    })
    returns_comparison.plot(kind='bar', ax=plt.gca())
    plt.title('Expected Returns Comparison')
    plt.ylabel('Expected Daily Return')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    
    plt.tight_layout()
    plt.savefig('portfolio_analysis.png')
    analysis_plot = plt.gcf()
    plt.close()
    
    # Calculate expected investment amounts
    total_investment = 10000  # $10,000
    optimal_investment = optimal_weights * total_investment
    mcap_investment = market_cap_weights * total_investment
    
    # Return all the relevant information
    return {
        'optimal_weights': optimal_weights,
        'market_cap_weights': market_cap_weights,
        'optimal_metrics': optimal_metrics,
        'mcap_metrics': mcap_metrics,
        'expected_returns': expected_returns,
        'adjusted_returns': adjusted_returns,
        'covariance_matrix': cov_matrix,
        'optimal_performance': optimal_performance,
        'mcap_performance': mcap_performance,
        'optimal_investment': optimal_investment,
        'mcap_investment': mcap_investment,
        'sentiment_adjustment': sentiment_adjustment
    }

def run_portfolio_optimization(predicted_prices_df, sentiment_df, actual_prices_df=None):
    """
    Main execution function to run the portfolio optimization and display results.
    
    Args:
        predicted_prices_df: DataFrame with predicted prices
        sentiment_df: DataFrame with sentiment data
        actual_prices_df: DataFrame with actual prices (optional)
        
    Returns:
        Dictionary containing all portfolio optimization results
    """
    # Run the optimization
    result = optimize_portfolio_with_constraints(
        predicted_prices_df,
        sentiment_df,
        market_caps=market_caps,
        min_weight=0.05,  # minimum 5% allocation to each stock
        risk_aversion=2.5,  # increased risk aversion for more diversification
        sentiment_weight=0.5  # increased sentiment impact
    )
    
    # Display the optimal weights
    print("Optimal Portfolio Weights:")
    for company, weight in result['optimal_weights'].items():
        print(f"{company}: {weight:.2%}")
    
    print("\nMarket Cap Weights:")
    for company, weight in result['market_cap_weights'].items():
        print(f"{company}: {weight:.2%}")
    
    # Display portfolio metrics
    print("\nOptimized Portfolio Metrics:")
    print(f"Daily Expected Return: {result['optimal_metrics']['expected_return']:.6f}")
    print(f"Annualized Expected Return: {result['optimal_metrics']['annualized_return']:.2%}")
    print(f"Daily Risk (Std Dev): {result['optimal_metrics']['risk']:.6f}")
    print(f"Annualized Risk: {result['optimal_metrics']['annualized_risk']:.2%}")
    print(f"Sharpe Ratio: {result['optimal_metrics']['sharpe_ratio']:.4f}")
    
    print("\nMarket Cap Portfolio Metrics:")
    print(f"Daily Expected Return: {result['mcap_metrics']['expected_return']:.6f}")
    print(f"Annualized Expected Return: {result['mcap_metrics']['annualized_return']:.2%}")
    print(f"Daily Risk (Std Dev): {result['mcap_metrics']['risk']:.6f}")
    print(f"Annualized Risk: {result['mcap_metrics']['annualized_risk']:.2%}")
    print(f"Sharpe Ratio: {result['mcap_metrics']['sharpe_ratio']:.4f}")
    
    # Show investment amounts
    print("\nOptimal Investment Amounts (Total $10,000):")
    for company, amount in result['optimal_investment'].items():
        print(f"{company}: ${amount:.2f}")
    
    # Compare predicted vs actual returns if actual data is provided
    if actual_prices_df is not None:
        # Get the date range from predicted prices
        start_date = predicted_prices_df['date'].min()
        end_date = predicted_prices_df['date'].max()
        
        comparison = compare_predicted_vs_actual(
            predicted_prices_df,
            actual_prices_df,
            start_date,
            end_date
        )
        
        print("\nPredicted vs Actual Returns:")
        for _, row in comparison.iterrows():
            print(f"{row['company']}: Predicted {row['predicted_return']:.2%}, Actual {row['actual_return']:.2%}")
        
        # Plot comparison
        plt.figure(figsize=(12, 6))
        comparison_melted = pd.melt(comparison, id_vars=['company'],
                                    value_vars=['predicted_return', 'actual_return'],
                                    var_name='Return Type', value_name='Return')
        
        sns.barplot(x='company', y='Return', hue='Return Type', data=comparison_melted)
        plt.title('Predicted vs Actual Returns')
        plt.xlabel('Company')
        plt.ylabel('Return (%)')
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig('return_comparison.png')
        plt.close()
    
    return result

def compare_portfolio_performance(performance_file1, performance_file2, output_file):
    """
    Compare performance of different portfolio strategies and save results.
    
    Args:
        performance_file1: Path to first performance CSV file
        performance_file2: Path to second performance CSV file
        output_file: Path to save the merged performance data
    """
    performance1 = pd.read_csv(performance_file1)
    performance2 = pd.read_csv(performance_file2)
    
    merged_performance = performance1.merge(
        performance2, on="Market Cap Portfolio", suffixes=("_chronos", "_with_sentiment")
    )
    merged_performance.to_csv(output_file, index=False)
    
    print(f"Merged performance data saved to {output_file}")

def main():
    """
    Main function to execute the portfolio optimization process.
    """
    # Define the tickers for analysis
    symbols = [
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
    
    # Define the time periods for analysis
    predict_start_date = '2025-01-01'
    predict_end_date = '2025-03-05'
    
    # Load sentiment data
    sentiment_df = load_sentiment_data('daily_sentiment_summary.csv', 'daily_sentiment_summary_2.csv')
    
    # Get actual prices for the prediction period to compare results
    actual_prices_df = get_actual_prices(symbols, predict_start_date, predict_end_date)
    
    # Run the optimization (assuming predicted_prices_df is available)
    # For demonstration purposes - in a real implementation, you would load or generate this data
    # base_result = run_portfolio_optimization(predicted_prices_df, None, actual_prices_df)
    # result = run_portfolio_optimization(predicted_prices_df, sentiment_df, actual_prices_df)
    
    # Compare different portfolio strategies
    # compare_portfolio_performance('performance.csv', 'performance_sentiment.csv', 'merged_performance.csv')
    
    print("Portfolio optimization script execution complete")

if __name__ == "__main__":
    main()
