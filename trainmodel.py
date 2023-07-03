import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import save_model

# Set the currency
symbol = "EURUSD=X"

# Define the date range for training data
start_date = pd.Timestamp('2004-01-01')
end_date = pd.Timestamp('2022-12-31')

# Download the currency data
forex_data = yf.download(symbol, start=start_date, end=end_date)

# Prepare the data for training
forex_close = forex_data['Close']
forex_data = forex_close.values.reshape(-1, 1)

# Normalize the data between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
forex_data = scaler.fit_transform(forex_data)

# Define the sequence length
seq_length = 30

step = 1  # Step size for creating sequences

X = []
y = []

for i in range(0, len(forex_data) - seq_length, step):
    X.append(forex_data[i:i + seq_length])
    y.append(forex_data[i + seq_length])

X = np.array(X)
y = np.array(y)

# Use all the data for training
X_train, y_train = X, y

# Build the LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(seq_length, 1)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Save the trained model
model.save("currency_model.h5")
