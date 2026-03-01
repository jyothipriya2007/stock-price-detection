import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

# 1. Download Data
ticker = "AAPL"  # Apple Inc.
data = yf.download(ticker, start="2020-01-01", end="2023-12-31")

# 2. Preprocessing
# We want to predict the 'Close' price based on the day number
data['Date_Ordinal'] = pd.to_datetime(data.index).map(pd.Timestamp.toordinal)
X = data[['Date_Ordinal']] # Features
y = data['Close']          # Target

# 3. Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predictions
predictions = model.predict(X_test)

# 6. Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='black', label='Actual Price')
plt.plot(X_test, predictions, color='blue', linewidth=3, label='Predicted Line')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Date (Ordinal)')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()