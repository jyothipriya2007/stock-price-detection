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