import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Define the list of tickers
tickers = ['TSLA', 'BND', 'SPY']

# Dictionary to store results for each ticker
results = {}

for ticker in tickers:
    print(f"\nProcessing {ticker}...\n")

    # Fetch data
    data = yf.download(ticker, start='2015-01-01', end='2024-10-31')
    data = data[['Close']]
    data.columns = [ticker]
    data = data.fillna(method='ffill')

    # Train-test split
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]

    # ARIMA model
    model_arima = ARIMA(train, order=(5, 1, 0))
    fitted_arima = model_arima.fit()
    forecast_arima = fitted_arima.forecast(steps=len(test))

    # SARIMA model
    model_sarima = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    fitted_sarima = model_sarima.fit()
    forecast_sarima = fitted_sarima.forecast(steps=len(test))

    # LSTM model
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)

    X_train, y_train = [], []
    for i in range(60, len(train_scaled)):
        X_train.append(train_scaled[i-60:i, 0])
        y_train.append(train_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    model_lstm = Sequential()
    model_lstm.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model_lstm.add(LSTM(50, return_sequences=False))
    model_lstm.add(Dense(25))
    model_lstm.add(Dense(1))
    model_lstm.compile(optimizer='adam', loss='mean_squared_error')
    model_lstm.fit(X_train, y_train, batch_size=32, epochs=10)

    X_test, y_test = [], []
    for i in range(60, len(test_scaled)):
        X_test.append(test_scaled[i-60:i, 0])
        y_test.append(test_scaled[i, 0])
    X_test = np.array(X_test).reshape((X_test.shape[0], X_test.shape[1], 1))

    predictions_lstm = model_lstm.predict(X_test)
    predictions_lstm = scaler.inverse_transform(predictions_lstm)

    # Evaluation metrics function
    def evaluate_model(true, pred):
        mae = mean_absolute_error(true, pred)
        rmse = mean_squared_error(true, pred, squared=False)
        mape = np.mean(np.abs((true - pred) / true)) * 100
        return mae, rmse, mape

    # Evaluate each model
    mae_arima, rmse_arima, mape_arima = evaluate_model(test, forecast_arima)
    mae_sarima, rmse_sarima, mape_sarima = evaluate_model(test, forecast_sarima)
    mae_lstm, rmse_lstm, mape_lstm = evaluate_model(test.values[60:], predictions_lstm)

    # Store results
    results[ticker] = {
        'ARIMA': {'MAE': mae_arima, 'RMSE': rmse_arima, 'MAPE': mape_arima},
        'SARIMA': {'MAE': mae_sarima, 'RMSE': rmse_sarima, 'MAPE': mape_sarima},
        'LSTM': {'MAE': mae_lstm, 'RMSE': rmse_lstm, 'MAPE': mape_lstm}
    }

    # Optional: Future forecasts
    future_forecast_arima = fitted_arima.get_forecast(steps=30).predicted_mean
    future_forecast_sarima = fitted_sarima.get_forecast(steps=30).predicted_mean

    last_60_days = test_scaled[-60:]
    X_future = last_60_days.reshape((1, last_60_days.shape[0], 1))
    future_forecast_lstm = []
    for _ in range(30):
        pred = model_lstm.predict(X_future)
        future_forecast_lstm.append(pred[0, 0])
        X_future = np.append(X_future[:, 1:, :], [[pred]], axis=1)
    future_forecast_lstm = scaler.inverse_transform(np.array(future_forecast_lstm).reshape(-1, 1))

# Display results
for ticker, metrics in results.items():
    print(f"\nResults for {ticker}:")
    for model_name, scores in metrics.items():
        print(f"  {model_name} - MAE: {scores['MAE']}, RMSE: {scores['RMSE']}, MAPE: {scores['MAPE']}")
