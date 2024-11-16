# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from scipy.optimize import minimize
import yfinance as yf

# Define the list of tickers
tickers = ['TSLA', 'BND', 'SPY']

# Dictionaries to store data and forecasts
data_dict = {}
forecast_dict_arima = {}
forecast_dict_sarima = {}
forecast_dict_lstm = {}

# Function for LSTM forecast
def forecast_lstm(train, test, steps):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)

    # Prepare data for LSTM
    X_train, y_train = [], []
    for i in range(60, len(train_scaled)):
        X_train.append(train_scaled[i-60:i])
        y_train.append(train_scaled[i])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, batch_size=32, epochs=50,  verbose=0)

    # Forecast future prices
    test_scaled = scaler.transform(test)
    X_test = np.array([test_scaled[:60]])  # Initial test sequence
    forecast = []
    for _ in range(steps):
        pred = model.predict(X_test, verbose=0)
        forecast.append(pred[0][0])
        pred_reshaped = np.reshape(pred, (1, 1, 1))
        X_test = np.append(X_test[:, 1:, :], pred_reshaped, axis=1)
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    return forecast.flatten()

# Process each ticker
for ticker in tickers:
    print(f"\nProcessing {ticker}...\n")

    # Fetch data
    data = yf.download(ticker, start='2015-01-01', end='2024-10-31')
    data = data[['Close']].rename(columns={'Close': ticker})
    data.fillna(method='ffill', inplace=True)
    data_dict[ticker] = data

    # Split data into train and test sets
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]

    # ARIMA forecast
    arima_model = ARIMA(train, order=(5, 1, 0))
    arima_fit = arima_model.fit()
    forecast_dict_arima[ticker] = arima_fit.forecast(steps=len(test))

    # SARIMA forecast
    sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    sarima_fit = sarima_model.fit(disp=False)
    forecast_dict_sarima[ticker] = sarima_fit.forecast(steps=len(test))

    # LSTM forecast
    lstm_forecast = forecast_lstm(train.values, test.values, steps=len(test))
    forecast_dict_lstm[ticker] = lstm_forecast

# Combine forecasts into DataFrames
forecast_df_arima = pd.DataFrame(forecast_dict_arima, index=data_dict[tickers[0]].index[-len(forecast_dict_arima[tickers[0]]):])
forecast_df_sarima = pd.DataFrame(forecast_dict_sarima, index=data_dict[tickers[0]].index[-len(forecast_dict_sarima[tickers[0]]):])
forecast_df_lstm = pd.DataFrame(forecast_dict_lstm, index=data_dict[tickers[0]].index[-len(forecast_dict_lstm[tickers[0]]):])

# Plot historical vs forecasts
plt.figure(figsize=(14, 7))
for ticker in tickers:
    plt.plot(data_dict[ticker], label=f"{ticker} Historical")
    plt.plot(forecast_df_arima[ticker], label=f"{ticker} ARIMA Forecast", linestyle='--')
    plt.plot(forecast_df_sarima[ticker], label=f"{ticker} SARIMA Forecast", linestyle=':')
    plt.plot(forecast_df_lstm[ticker], label=f"{ticker} LSTM Forecast", linestyle='-.')
plt.title("Forecasted vs Historical Prices for TSLA, BND, SPY")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

# Step 4: Calculate daily returns
forecasted_daily_returns = forecast_df_lstm.pct_change().dropna()

# Calculate annualized returns and covariance matrix
annualized_forecasted_return = forecasted_daily_returns.mean() * 252
cov_matrix_forecasted = forecasted_daily_returns.cov() * 252

print("\nAnnualized Forecasted Return:\n", annualized_forecasted_return)
print("\nForecasted Covariance Matrix:\n", cov_matrix_forecasted)

# Step 5: Optimize portfolio allocation
initial_weights = np.array([1/3] * len(tickers))

# Define negative Sharpe Ratio function
def neg_sharpe(weights):
    portfolio_return = np.dot(weights, annualized_forecasted_return)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix_forecasted, weights)))
    return -portfolio_return / portfolio_volatility

# Constraints and bounds
constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
bounds = tuple((0, 1) for _ in range(len(initial_weights)))

# Run optimization
optimized_results = minimize(neg_sharpe, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
optimized_weights = optimized_results.x

print("\nOptimized Weights:\n", dict(zip(tickers, optimized_weights)))

# Portfolio performance
optimized_portfolio_return = np.dot(optimized_weights, annualized_forecasted_return)
optimized_portfolio_volatility = np.sqrt(np.dot(optimized_weights.T, np.dot(cov_matrix_forecasted, optimized_weights)))
optimized_sharpe_ratio = optimized_portfolio_return / optimized_portfolio_volatility

print(f"\nOptimized Portfolio Return: {optimized_portfolio_return:.4f}")
print(f"Optimized Portfolio Volatility: {optimized_portfolio_volatility:.4f}")
print(f"Optimized Sharpe Ratio: {optimized_sharpe_ratio:.4f}")
