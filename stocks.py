import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Data Collection
symbol = 'AAPL'
data = yf.download(symbol, start='2010-01-01', end='2020-12-31')

# Data Preprocessing
features = ['Open', 'High', 'Low', 'Volume']
X = data[features].values
y = data['Close'].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Building
model = Sequential()
model.add(Dense(64, input_dim=4, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Model Compilation
model.compile(optimizer='adam', loss='mse')

# Model Training
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Model Evaluation
y_pred = model.predict(X_test)
mse = np.mean((y_pred - y_test) ** 2)
print('Mean Squared Error:', mse)

# Prediction (assuming next_day_pred is an array with multiple predictions)
next_day_pred = np.array([10, 12, 15, 18])  # Replace with your actual prediction
next_day_pred = next_day_pred.reshape(1, -1)  # Reshape to (1, 4)
print("Reshaped next_day_pred:", next_day_pred)
print("Predicted Price (inverse transformed):", scaler.inverse_transform(next_day_pred))
