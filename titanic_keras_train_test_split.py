# -----------------------------------------------------
# Neural Network (Keras) - Titanic Survival Prediction
#
# New concepts added:
#   1. Train/Test Split
#   2. Validation Data
#   3. Dropout Layer
#   4. Model Evaluation
#
# Features used:
#   1. pclass
#   2. gender
#   3. age
#
# Architecture:
#
# Input(3)
#    ↓
# Dense(6, ReLU)
#    ↓
# Dropout(0.2)
#    ↓
# Dense(4, ReLU)
#    ↓
# Output(1, Sigmoid)
# -----------------------------------------------------

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import layers


# -----------------------------------------------------
# STEP 1: Load dataset
# -----------------------------------------------------
df = pd.read_csv("titanic.csv")

# Keep required columns
df = df[[
    "survived",
    "pclass",
    "gender",
    "age"
]].dropna()


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
# STEP 3: Prepare features and labels
# -----------------------------------------------------
X = df[[
    "pclass",
    "gender",
    "age"
]].values

y = df["survived"].values


# -----------------------------------------------------
# STEP 4: Feature scaling
# -----------------------------------------------------

# Scale passenger class
X[:, 0] = X[:, 0] / 3

# Scale age
X[:, 2] = X[:, 2] / 80


# -----------------------------------------------------
# STEP 5: Train/Test Split
# -----------------------------------------------------
# 80% training data
# 20% testing data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------------------------------
# STEP 6: Build Neural Network
# -----------------------------------------------------
model = keras.Sequential([

    # Hidden Layer 1
    layers.Dense(
        6,
        activation='relu',
        input_shape=(3,)
    ),

    # Dropout layer
    # Randomly disables 20% neurons during training
    # Helps reduce overfitting
    layers.Dropout(0.2),

    # Hidden Layer 2
    layers.Dense(
        4,
        activation='relu'
    ),

    # Output Layer
    layers.Dense(
        1,
        activation='sigmoid'
    )
])


# -----------------------------------------------------
# STEP 7: Compile Model
# -----------------------------------------------------
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# -----------------------------------------------------
# STEP 8: Train Model
# -----------------------------------------------------
# validation_split=0.2
#
# 20% of TRAINING data is used for validation
# Validation helps monitor generalization

history = model.fit(

    X_train,
    y_train,

    epochs=50,
    batch_size=16,

    validation_split=0.2
)


# -----------------------------------------------------
# STEP 9: Model Summary
# -----------------------------------------------------
print("\n--- Model Summary ---")

model.summary()


# -----------------------------------------------------
# STEP 10: Evaluate on Test Data
# -----------------------------------------------------
print("\n--- Test Evaluation ---")

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("Test Loss     :", round(loss, 4))
print("Test Accuracy :", round(accuracy, 4))


# -----------------------------------------------------
# STEP 11: Testing / Predictions
# -----------------------------------------------------
print("\n--- Sample Predictions ---")


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


# -----------------------------------------------------
# STEP 12: Show Training History
# -----------------------------------------------------
print("\n--- Training History ---")

print("Final Training Accuracy :",
      round(history.history['accuracy'][-1], 4))

print("Final Validation Accuracy :",
      round(history.history['val_accuracy'][-1], 4))