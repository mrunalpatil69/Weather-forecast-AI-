# Weather-forecast-AI-
It's an AI for weather forecasting 


@@ -1 +1,52 @@
# Mrunal-patil
Creating a sophisticated weather forecasting AI involves a complex process and usually requires substantial computing resources and access to extensive datasets. It's not feasible to provide a comprehensive code example in this chat. However, I can provide a simplified example using a recurrent neural network (RNN) for time series forecasting. Note that this is a basic example and may not be suitable for accurate weather prediction.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Assuming you have a DataFrame 'weather_data' with columns like 'temperature', 'humidity', etc.
# Ensure your data is preprocessed and includes a target variable (e.g., 'next_day_temperature')

# Feature scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(weather_data)

# Define the number of time steps for the LSTM model
time_steps = 5  # Adjust as needed based on your data

# Create sequences for training
X, y = [], []
for i in range(len(scaled_data) - time_steps):
    X.append(scaled_data[i:i + time_steps, :-1])
    y.append(scaled_data[i + time_steps, -1])

X, y = np.array(X), np.array(y)

# Reshape input for LSTM model
X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=50, batch_size=32)

# Make predictions on new data (adjust as needed)
test_input = scaled_data[-time_steps:]
test_input = np.reshape(test_input, (1, test_input.shape[0], test_input.shape[1]))
predicted_output = model.predict(test_input)

# Inverse transform to get the original scale
predicted_output = scaler.inverse_transform(np.array([[0] * (scaled_data.shape[1] - 1) + [predicted_output[0, 0]]]))

print("Predicted Next Day Temperature:", predicted_output[0, -1])
