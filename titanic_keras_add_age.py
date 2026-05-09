# -----------------------------------------------------
# Neural Network (Keras) - Titanic Survival Prediction
# Features used:
#   1. pclass
#   2. gender
#   3. age
#
# Goal:
# Predict whether a passenger survived (0/1)
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
# male   → 0
# female → 1
df["gender"] = df["gender"].map({
    "male": 0,
    "female": 1
})


# -----------------------------------------------------
# STEP 3: Prepare features (X) and labels (y)
# -----------------------------------------------------
# Features:
# pclass
# gender
# age

X = df[["pclass", "gender", "age"]].values
y = df["survived"].values


# -----------------------------------------------------
# STEP 4: Feature scaling
# -----------------------------------------------------
# Neural networks train better when values are
# roughly in similar ranges

# Scale passenger class (1-3) → 0-1
X[:, 0] = X[:, 0] / 3

# Scale age roughly to 0-1
# Titanic max age is around 80
X[:, 2] = X[:, 2] / 80


# -----------------------------------------------------
# STEP 5: Build Neural Network
# -----------------------------------------------------
# Input:
#   3 features
#
# Structure:
#   Input → Hidden Layer → Output

model = keras.Sequential([

    # Hidden layer
    layers.Dense(
        6,
        activation='relu',
        input_shape=(3,)
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
# STEP 8: Testing / Predictions
# -----------------------------------------------------
print("\n--- Testing ---")


# -----------------------------------------------------
# Example 1:
# 1st class female, age 25
# -----------------------------------------------------
# pclass = 1 → 1/3
# gender = female → 1
# age = 25 → 25/80

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


# -----------------------------------------------------
# Example 2:
# 3rd class male, age 40
# -----------------------------------------------------
test = np.array([
    [1/3, 0, 40/80]
])

p = model.predict(test)

print(
    "3rd class male age 40 →",
    round(float(p[0][0]), 3),
    "→",
    int(p[0][0] >= 0.5)
)


# -----------------------------------------------------
# Example 3:
# 2nd class male, age 10
# -----------------------------------------------------
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