import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import datetime

# 1. Download Data
ticker = "AAPL"
data = yf.download(ticker, start="2026-02-02", end="2026-03-03")

# 2. Preprocessing
# We use all available data to train since we are heading into the unknown
data['Date_Ordinal'] = pd.to_datetime(data.index).map(pd.Timestamp.toordinal)
X = data[['Date_Ordinal']] 
y = data['Close']

# 3. Train on EVERYTHING (No split, we want maximum data for the future)
model = LinearRegression()
model.fit(X, y)

# 4. Generate Future Dates (e.g., next 30 days)
last_date = data.index[-1]
future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, 61)]
future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)

# 5. Future Predictions
future_preds = model.predict(future_ordinals)

# 6. Visualization
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Historical Price', color='black')
plt.plot(future_dates, future_preds, label='Future Forecast', color='red', linestyle='--')

plt.title(f'{ticker} 60-Day Future Forecast')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()  