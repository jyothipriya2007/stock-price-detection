import numpy as np
from sklearn.linear_model import LinearRegression
import datetime

def train_model(data):
    X = data[['Date_Ordinal']]
    y = data['Close']
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_future(model, last_date, days=60):
    future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, days+1)]
    future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    future_preds = model.predict(future_ordinals)
    return future_dates, future_preds