# GMF-Financial-Analysis
## Workflow
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

#Step 4: Seasonality and Trends
Decompose the time series into trend, seasonality, and residual components.
```
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose TSLA's closing prices as an example
decomposition = seasonal_decompose(data['TSLA'], model='multiplicative', period=252)
decomposition.plot()
plt.show()

```
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
#Key Insights
Overall Direction of Tesla’s Stock Price: The trend component from the decomposition will indicate the direction of TSLA’s stock over the years.

Fluctuations in Daily Returns: Volatility bands and outliers show periods of high volatility, which can indicate key events affecting Tesla, bond stability (BND), and broader market trends (SPY).

Risk-Adjusted Return: The Sharpe Ratio assesses the risk-adjusted return of each asset, with higher values indicating more efficient returns per unit of risk.


