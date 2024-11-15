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

Step 4: 


