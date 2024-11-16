import yfinance as yf
import pandas as pd

# Define the tickers for TSLA, BND, and SPY
tickers = ["TSLA", "BND", "SPY"]

# Download data from Yahoo Finance for the selected tickers
data = yf.download(tickers, start="2015-01-01", end="2023-01-01")['Adj Close']
data.columns = ['TSLA', 'BND', 'SPY']  # Rename columns for clarity
data.head()
