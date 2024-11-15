# GMF-Financial-Analysis
## Task 1 : Preprocess and Explore the Data Workflow
#step - 1: Fetching the financial data
To fetch historical financial data for TSLA, BND, and SPY from January 1, 2015, to October 31, 2024, you can use the yfinance library in Python. Here’s how to download the data for that specific date range.
```
pip install yfinance
```
Now, here’s the Python code to fetch the data:
```
import yfinance as yf
import pandas as pd

# Define the tickers for TSLA, BND, and SPY
tickers = ["TSLA", "BND", "SPY"]

# Download data from Yahoo Finance for the selected tickers
data = yf.download(tickers, start="2015-01-01", end="2024-10-31")['Adj Close']

# Rename columns for easier access
data.columns = ['TSLA', 'BND', 'SPY']  
print(data.head())  # Display the first few rows of data
```
| Date                     | TSLA           | BND        | SPY      |
|--------------------------|----------------|------------|----------|
| 2015-01-02 00:00:00+00:00| 63.358723      | 173.173767 | 14.620667|
| 2015-01-05 00:00:00+00:00| 63.542759      | 170.046310 | 14.006000|
| 2015-01-06 00:00:00+00:00| 63.726715      | 168.444672 | 14.085333|

#step 2: Data cleaning and understanding

1, check Data Types and Missing values
```
print(data.info())  # Check data types and missing values
print(data.describe())  # Summary statistics

```
2, Handle Missing Values
```
# Forward fill and backward fill for any missing values
data.fillna(method='ffill', inplace=True)
data.fillna(method='bfill', inplace=True)

```
3, Normalize or scale Data
This can be useful for machine learning, but since we’re primarily analyzing trends, normalization may be optional. For machine learning, use the following:
```
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
```
#Step 3: Exploratory Data Analysis (EDA)
1, Visualize Closing Prices Over Time
```
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 7))
for col in data.columns:
    plt.plot(data.index, data[col], label=col)
plt.title("Closing Prices Over Time")
plt.xlabel("Date")
plt.ylabel("Adjusted Close Price")
plt.legend()
plt.show()

```
![Untitled](https://github.com/user-attachments/assets/383cab1d-c78b-4eaf-97e2-671d19203cf1)

2, Calculate and Plot Daily Percentage Change
```
daily_returns = data.pct_change().dropna()  # Daily percentage change
daily_returns.plot(figsize=(14, 7), title="Daily Percentage Change", xlabel="Date", ylabel="Daily Return (%)")
plt.show()

```
![Untitled](https://github.com/user-attachments/assets/62d4ce1b-cabb-4466-b9e0-b61677c0bc00)
3, Analyze Volatility

    -Rolling Means and Standard Deviations
```
    # Calculate 30-day rolling means and standard deviations
rolling_mean = data.rolling(window=30).mean()
rolling_std = data.rolling(window=30).std()

plt.figure(figsize=(14, 7))
for col in data.columns:
    plt.plot(rolling_mean.index, rolling_mean[col], label=f'{col} 30-Day Mean')
    plt.fill_between(rolling_std.index, (rolling_mean[col] - rolling_std[col]), 
                     (rolling_mean[col] + rolling_std[col]), alpha=0.1)
plt.title("30-Day Rolling Means and Volatility Bands")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

 ```
![Untitled](https://github.com/user-attachments/assets/4412a4b5-72f5-4be3-88ba-56c688b38d81)
4, Outlier Detection (Days with High/Low Returns)

```
# Identify outliers based on daily returns
outliers = daily_returns[(daily_returns > daily_returns.mean() + 3*daily_returns.std()) |
                         (daily_returns < daily_returns.mean() - 3*daily_returns.std())]

plt.figure(figsize=(14, 7))
for col in outliers.columns:
    plt.scatter(outliers.index, outliers[col], label=f'{col} Outliers', alpha=0.5)
plt.title("Days with Unusually High/Low Returns")
plt.xlabel("Date")
plt.ylabel("Return (%)")
plt.legend()
plt.show()

```
![Untitled](https://github.com/user-attachments/assets/f4177b18-5903-4e91-8e0f-37f844a6607d)


#Step 4: Seasonality and Trends
Decompose the time series into trend, seasonality, and residual components.
```
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose TSLA's closing prices as an example
decomposition = seasonal_decompose(data['TSLA'], model='multiplicative', period=252)
decomposition.plot()
plt.show()

```
![Untitled](https://github.com/user-attachments/assets/1e2ac158-610d-4d00-a3c3-004b55c70033)
#Step 5: Volatility Analysis and Key Insights

1,Calculate Value at Risk (VaR) and Sharpe Ratio

    -VaR Calculation
```
# 5% VaR assuming normal distribution
var_95 = daily_returns.quantile(0.05)
print("95% VaR for each asset:\n", var_95)

```
    -Sharpe Ratio
```
# Assume a risk-free rate of 0 for simplicity
sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * (252**0.5))
print("Sharpe Ratio for each asset:\n", sharpe_ratio)
```
#Key Insights:

Overall Direction of Tesla’s Stock Price: The trend component from the decomposition will indicate the direction of TSLA’s stock over the years.

Fluctuations in Daily Returns: Volatility bands and outliers show periods of high volatility, which can indicate key events affecting Tesla, bond stability (BND), and broader market trends (SPY).

Risk-Adjusted Return: The Sharpe Ratio assesses the risk-adjusted return of each asset, with higher values indicating more efficient returns per unit of risk.

#Task 2: Develop Time Series Forecasting Models

This task involves building a time series forecasting model to predict all assets's future stock prices. Below are the step-by-step implementation to develop, evaluate, and refine a forecasting model using common techniques such as ARIMA, SARIMA, or LSTM.

#Step 1: Forecast Future Prices for All Assets (TSLA, BND, SPY)

Use your chosen model (e.g., ARIMA, SARIMA, or LSTM) for each asset individually. Here’s a rough example for three assets:
```
# Forecasting prices for TSLA, BND, SPY (repeat similar code for each asset)

# Example code for TSLA
tsla_forecast = forecast_lstm(tsla_data)  # Use your LSTM or ARIMA model here

# Example code for BND and SPY
bnd_forecast = forecast_sarima(bnd_data)  # Use SARIMA model here
spy_forecast = forecast_arima(spy_data)   # Use ARIMA model here

# Combine into a DataFrame
forecast_data = pd.DataFrame({
    'TSLA': tsla_forecast,
    'BND': bnd_forecast,
    'SPY': spy_forecast
})

```
#Step 2: Compile Forecasted Data into a Portfolio DataFrame
```
# Combine historical data and forecast data
historical_data = pd.DataFrame({
    'TSLA': tsla_data,
    'BND': bnd_data,
    'SPY': spy_data
})
combined_df = pd.concat([historical_data, forecast_data], axis=0)

# Visualize the combined data to check trends
combined_df.plot(figsize=(14, 7))
plt.title("Historical and Forecasted Prices for TSLA, BND, SPY")
plt.show()

```
#Step 3: Calculate Expected Returns and Volatility for Each Asset

Use the forecasted prices to calculate daily returns and then annualize them.
```
# Calculate daily returns from forecast data
forecasted_daily_returns = forecast_data.pct_change()

# Annualized returns
annualized_forecasted_return = forecasted_daily_returns.mean() * 252
cov_matrix_forecasted = forecasted_daily_returns.cov() * 252

print("Annualized Forecasted Return:", annualized_forecasted_return)
print("Forecasted Covariance Matrix:", cov_matrix_forecasted)

```
Step 4: Define Initial Weights and Optimize Portfolio Weights
Use optimization to find the weight distribution that maximizes the Sharpe ratio based on the forecasted data.
```
from scipy.optimize import minimize

# Set initial weights (e.g., equal weighting or any other initial guess)
initial_weights = [1/3, 1/3, 1/3]

# Define negative Sharpe Ratio function for optimization
def neg_sharpe(weights):
    portfolio_return = np.dot(weights, annualized_forecasted_return)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix_forecasted, weights)))
    return -portfolio_return / portfolio_volatility

# Constraints and bounds
constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
bounds = tuple((0, 1) for _ in range(len(initial_weights)))

# Run optimization
optimized_results = minimize(neg_sharpe, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
optimized_weights = optimized_results.x

print("Optimized Weights:", optimized_weights)

```
#Step 5: Calculate Portfolio Performance Metrics and Visualize Results

After optimizing the weights, calculate and visualize the expected portfolio returns, volatility, and Sharpe ratio.
```
# Calculate optimized portfolio performance
optimized_portfolio_return = np.dot(optimized_weights, annualized_forecasted_return)
optimized_portfolio_volatility = np.sqrt(np.dot(optimized_weights.T, np.dot(cov_matrix_forecasted, optimized_weights)))
optimized_sharpe_ratio = optimized_portfolio_return / optimized_portfolio_volatility

print("Optimized Portfolio Return:", optimized_portfolio_return)
print("Optimized Portfolio Volatility:", optimized_portfolio_volatility)
print("Optimized Sharpe Ratio:", optimized_sharpe_ratio)

# Plot cumulative returns
cumulative_returns = (1 + forecasted_daily_returns).cumprod()
optimized_cumulative_returns = (1 + forecasted_daily_returns.dot(optimized_weights)).cumprod()

plt.figure(figsize=(14, 7))
for asset in ['TSLA', 'BND', 'SPY']:
    plt.plot(cumulative_returns[asset], label=f"{asset} Cumulative Return")
plt.plot(optimized_cumulative_returns, label="Optimized Portfolio", linestyle='--')
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.title("Cumulative Return for Each Asset and Optimized Portfolio")
plt.legend()
plt.show()

```
#Step 6: Interpret Results and Make Portfolio Adjustments
Based on the optimized weights and forecast analysis:

Expected Return and Volatility: Higher return and Sharpe ratio indicate potential for growth with minimized risk.
Weight Adjustments: Increase weights in stable assets like BND if Tesla (TSLA) shows high forecasted volatility.

