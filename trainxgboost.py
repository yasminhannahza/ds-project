import pickle
import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Download the currency data
start_date = pd.Timestamp('2013-01-01')
end_date = pd.Timestamp.today()  # Updated to current date
forex_data = yf.download('EURUSD=X', start=start_date, end=end_date)

# Prepare the data for training
forex_close = forex_data['Close']

# Define the sequence length
seq_length = 30

step = 1  # Step size for creating sequences

X = []
y = []

for i in range(0, len(forex_close) - seq_length, step):
    X.append(forex_close[i:i + seq_length])
    y.append(forex_close[i + seq_length])

X = np.array(X)
y = np.array(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Create and train the XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.1, random_state=123)
model.fit(X_train, y_train)

# Save the trained model
with open('xgboost_model.h5', 'wb') as file:
    pickle.dump(model, file)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print(f"RMSE on the test set: {rmse}")
print(f"MAE on the test set: {mae}")

# Perform predictions using the model
num_days = 7  # Number of days for future prediction
X_pred = forex_close[-seq_length:].values
future_predictions = []

for _ in range(num_days):
    prediction = model.predict(X_pred.reshape(1, seq_length))
    future_predictions.append(prediction[0])
    X_pred = np.append(X_pred[1:], prediction)

# Create the timeline for the future predictions (using business days)
last_date = forex_close.index[-1]
future_dates = pd.bdate_range(start=last_date + pd.DateOffset(days=1), periods=num_days)

# Print the future predictions
print("Future Predictions:")
for date, prediction in zip(future_dates, future_predictions):
    print(f"{date}: {prediction}")
