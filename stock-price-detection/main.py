import yfinance as yf
import pandas as pd

def get_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data['Date_Ordinal'] = pd.to_datetime(data.index).map(pd.Timestamp.toordinal)
    return data
import numpy as np
import sklearn.linear_model
import datetime

def train_model(data):
    X = data[['Date_Ordinal']]
    y = data['Close']
    model = sklearn.linear_model.LinearRegression()
    model.fit(X, y)
    return model

def predict_future(model, last_date, days=60):
    future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, days+1)]
    future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    future_preds = model.predict(future_ordinals)
    return future_dates, future_preds
import matplotlib.pyplot as plt

def plot_forecast(data, future_dates, future_preds, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Historical Price', color='black')
    plt.plot(future_dates, future_preds, label='Future Forecast', color='red', linestyle='--')
    plt.title(f'{ticker} {len(future_dates)}-Day Future Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()
    from src.data_loader import get_stock_data
from src.model import train_model, predict_future
from src.visualize import plot_forecast

# Parameters
ticker = "AAPL"
start = "2024-01-01"
end = "2024-12-31"

# Workflow
data = get_stock_data(ticker, start, end)
model = train_model(data)
future_dates, future_preds = predict_future(model, data.index[-1], days=60)
plot_forecast(data, future_dates, future_preds, ticker)
