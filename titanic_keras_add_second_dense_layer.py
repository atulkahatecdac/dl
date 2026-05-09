# -----------------------------------------------------
# Neural Network (Keras) - Titanic Survival Prediction
# Features used:
#   1. pclass
#   2. gender
#   3. age
#
# Architecture:
# Input → Dense Layer 1 → Dense Layer 2 → Output
# -----------------------------------------------------

import pandas as pd
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers


# -----------------------------------------------------
# STEP 1: Load dataset
# -----------------------------------------------------
df = pd.read_csv("titanic.csv")

# Keep required columns
df = df[["survived", "pclass", "gender", "age"]].dropna()


# -----------------------------------------------------
# STEP 2: Convert categorical data → numeric
# -----------------------------------------------------
df["gender"] = df["gender"].map({
    "male": 0,
    "female": 1
})


# -----------------------------------------------------
# STEP 3: Prepare features and labels
# -----------------------------------------------------
X = df[["pclass", "gender", "age"]].values
y = df["survived"].values


# -----------------------------------------------------
# STEP 4: Feature scaling
# -----------------------------------------------------

# Scale passenger class
X[:, 0] = X[:, 0] / 3

# Scale age
X[:, 2] = X[:, 2] / 80


# -----------------------------------------------------
# STEP 5: Build Neural Network
# -----------------------------------------------------
# Architecture:
#
# Input (3 features)
#        ↓
# Dense Layer 1 (6 neurons)
#        ↓
# Dense Layer 2 (4 neurons)
#        ↓
# Output Layer (1 neuron)

model = keras.Sequential([

    # First hidden layer
    layers.Dense(
        6,
        activation='relu',
        input_shape=(3,)
    ),

    # Second hidden layer
    layers.Dense(
        4,
        activation='relu'
    ),

    # Output layer
    layers.Dense(
        1,
        activation='sigmoid'
    )
])


# -----------------------------------------------------
# STEP 6: Compile model
# -----------------------------------------------------
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# -----------------------------------------------------
# STEP 7: Train model
# -----------------------------------------------------
model.fit(
    X,
    y,
    epochs=50,
    batch_size=16
)


# -----------------------------------------------------
# STEP 8: Model Summary
# -----------------------------------------------------
print("\n--- Model Summary ---")
model.summary()


# -----------------------------------------------------
# STEP 9: Testing / Predictions
# -----------------------------------------------------
print("\n--- Testing ---")


# Example 1
# 1st class female age 25
test = np.array([
    [1/3, 1, 25/80]
])

p = model.predict(test)

print(
    "1st class female age 25 →",
    round(float(p[0][0]), 3),
    "→",
    int(p[0][0] >= 0.5)
)


# Example 2
# 3rd class male age 40
test = np.array([
    [1, 0, 40/80]
])

p = model.predict(test)

print(
    "3rd class male age 40 →",
    round(float(p[0][0]), 3),
    "→",
    int(p[0][0] >= 0.5)
)


# Example 3
# 2nd class male age 10
test = np.array([
    [2/3, 0, 10/80]
])

p = model.predict(test)

print(
    "2nd class male age 10 →",
    round(float(p[0][0]), 3),
    "→",
    int(p[0][0] >= 0.5)
)