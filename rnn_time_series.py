# -----------------------------------------------------
# RNN Time Series Forecasting
# Using Real VIX Dataset
#
# Goal:
# Predict next VIX value
# using previous values
# -----------------------------------------------------

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from tensorflow import keras
from tensorflow.keras import layers


# -----------------------------------------------------
# STEP 1: Load Dataset
# -----------------------------------------------------

url = "https://raw.githubusercontent.com/datasets/finance-vix/main/data/vix-daily.csv"

df = pd.read_csv(url)

print("\n--- Dataset ---")
print(df.head())


# -----------------------------------------------------
# STEP 2: Keep Only VIX Close Price
# -----------------------------------------------------

prices = df["CLOSE"].values

print("\nTotal records:", len(prices))


# -----------------------------------------------------
# STEP 3: Normalize Using MinMaxScaler
# -----------------------------------------------------
#
# MinMaxScaler scales values between 0 and 1
#
# Formula:
#
# X_scaled =
# (X - X_min) / (X_max - X_min)
#

scaler = MinMaxScaler()

# sklearn expects 2D data
prices = prices.reshape(-1, 1)

prices_scaled = scaler.fit_transform(prices)


print("\n--- First 5 Scaled Values ---")
print(prices_scaled[:5])


# -----------------------------------------------------
# STEP 4: Create Sliding Windows
# -----------------------------------------------------
#
# Use previous 5 days
# to predict next day
#

window_size = 5

X = []
y = []

for i in range(len(prices_scaled) - window_size):

    X.append(

        prices_scaled[i:i + window_size] # prices of previous 5 days
    )

    y.append(

        prices_scaled[i + window_size] # price of next day
    )


X = np.array(X) # convert to numpy array
y = np.array(y)


# -----------------------------------------------------
# STEP 5: Reshape for RNN
# -----------------------------------------------------
#
# Current shape:
# (samples, timesteps, 1)
#
# because each day has:
# one feature = VIX value
#

print("\nX shape:", X.shape) # samples, timesteps, 1
print("y shape:", y.shape) # samples


# -----------------------------------------------------
# STEP 6: Split Train/Test
# -----------------------------------------------------

split = int(len(X) * 0.8)

X_train = X[:split]
y_train = y[:split]

X_test = X[split:]
y_test = y[split:]


# -----------------------------------------------------
# STEP 7: Build RNN Model
# -----------------------------------------------------

model = keras.Sequential([

    layers.SimpleRNN(

        32, # number of hidden units

        activation='tanh', # tanh = hyperbolic tangent, used for RNN

        input_shape=(window_size,1) # input shape
    ),

    layers.Dense(1) # output layer
])


# -----------------------------------------------------
# STEP 8: Compile Model
# -----------------------------------------------------

model.compile(

    optimizer='adam',

    loss='mse'
)


# -----------------------------------------------------
# STEP 9: Train Model
# -----------------------------------------------------

model.fit(

    X_train,
    y_train,

    epochs=20,

    batch_size=32,

    validation_data=(X_test, y_test)
)


# -----------------------------------------------------
# STEP 10: Evaluate Model
# -----------------------------------------------------

loss = model.evaluate( # evaluate on test data

    X_test,
    y_test
)

print("\nTest Loss:", loss)


# -----------------------------------------------------
# STEP 11: Predict Next Value
# -----------------------------------------------------
#
# Use latest 5 days
#

latest_sequence = prices_scaled[-window_size:] # last 5 days

print("\nLatest Sequence:")
print(latest_sequence)


# Shape becomes:
# (1,5,1)

test = latest_sequence.reshape(

    1,
    window_size,
    1
)


prediction_scaled = model.predict(test)


# -----------------------------------------------------
# STEP 12: Convert Back to Original Scale
# -----------------------------------------------------

prediction = scaler.inverse_transform(

    prediction_scaled
)


print(

    "\nPredicted Next VIX Value:",
    round(float(prediction[0][0]), 2)
)