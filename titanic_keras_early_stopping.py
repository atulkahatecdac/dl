# -----------------------------------------------------
# Neural Network (Keras) - Titanic Survival Prediction
#
# PROFESSIONAL VERSION
#
# New concepts added:
#   1. StandardScaler
#   2. Train/Test Split
#   3. Validation Data
#   4. Dropout
#   5. EarlyStopping
#   6. ModelCheckpoint
#   7. Training History
#
# Features used:
#   1. pclass
#   2. gender
#   3. age
#   4. fare
#
# Architecture:
#
# Input(4)
#    ↓
# Dense(8, ReLU)
#    ↓
# Dropout(0.3)
#    ↓
# Dense(4, ReLU)
#    ↓
# Output(1, Sigmoid)
# -----------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)


# -----------------------------------------------------
# STEP 1: Load dataset
# -----------------------------------------------------
df = pd.read_csv("titanic.csv")

# Keep required columns
df = df[[
    "survived",
    "pclass",
    "gender",
    "age",
    "fare"
]].dropna()


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
X = df[[
    "pclass",
    "gender",
    "age",
    "fare"
]].values

y = df["survived"].values


# -----------------------------------------------------
# STEP 4: Train/Test Split
# -----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,
    random_state=42
)


# -----------------------------------------------------
# STEP 5: Feature Scaling
# -----------------------------------------------------
# StandardScaler:
#
# value = (value - mean) / std_dev
#
# Makes training more stable

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

# IMPORTANT:
# Use SAME scaler on test data
X_test = scaler.transform(X_test)


# -----------------------------------------------------
# STEP 6: Build Neural Network
# -----------------------------------------------------
model = keras.Sequential([

    # Hidden Layer 1
    layers.Dense(
        8,
        activation='relu',
        input_shape=(4,)
    ),

    # Dropout
    layers.Dropout(0.3),

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
# STEP 8: Callbacks
# -----------------------------------------------------

# EarlyStopping:
# Stop training if validation loss
# stops improving

early_stop = EarlyStopping(

    monitor='val_loss',

    patience=10,

    restore_best_weights=True
)


# ModelCheckpoint:
# Save best model during training

checkpoint = ModelCheckpoint(

    "best_model.keras",

    monitor='val_loss',

    save_best_only=True,

    verbose=1
)


# -----------------------------------------------------
# STEP 9: Train Model
# -----------------------------------------------------
history = model.fit(

    X_train,
    y_train,

    epochs=100,

    batch_size=16,

    validation_split=0.2,

    callbacks=[
        early_stop,
        checkpoint
    ]
)


# -----------------------------------------------------
# STEP 10: Model Summary
# -----------------------------------------------------
print("\n--- Model Summary ---")

model.summary()


# -----------------------------------------------------
# STEP 11: Evaluate on Test Data
# -----------------------------------------------------
print("\n--- Test Evaluation ---")

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("Test Loss     :", round(loss, 4))
print("Test Accuracy :", round(accuracy, 4))


# -----------------------------------------------------
# STEP 12: Sample Predictions
# -----------------------------------------------------
print("\n--- Sample Predictions ---")

#
# IMPORTANT:
# Input must be scaled using SAME scaler
#

# Example 1
# 1st class female age 25 fare 100

test = np.array([
    [1, 1, 25, 100]
])

test = scaler.transform(test)

p = model.predict(test)

print(
    "1st class female age 25 fare 100 →",
    round(float(p[0][0]), 3),
    "→",
    int(p[0][0] >= 0.5)
)


# Example 2
# 3rd class male age 40 fare 10

test = np.array([
    [3, 0, 40, 10]
])

test = scaler.transform(test)

p = model.predict(test)

print(
    "3rd class male age 40 fare 10 →",
    round(float(p[0][0]), 3),
    "→",
    int(p[0][0] >= 0.5)
)


# Example 3
# 2nd class male age 10 fare 25

test = np.array([
    [2, 0, 10, 25]
])

test = scaler.transform(test)

p = model.predict(test)

print(
    "2nd class male age 10 fare 25 →",
    round(float(p[0][0]), 3),
    "→",
    int(p[0][0] >= 0.5)
)


# -----------------------------------------------------
# STEP 13: Training History
# -----------------------------------------------------
print("\n--- Final Metrics ---")

print(
    "Final Training Accuracy :",
    round(history.history['accuracy'][-1], 4)
)

print(
    "Final Validation Accuracy :",
    round(history.history['val_accuracy'][-1], 4)
)


# -----------------------------------------------------
# STEP 14: Plot Accuracy Graph
# -----------------------------------------------------
plt.figure(figsize=(8, 5))

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title("Training vs Validation Accuracy")

plt.legend()

plt.show()


# -----------------------------------------------------
# STEP 15: Plot Loss Graph
# -----------------------------------------------------
plt.figure(figsize=(8, 5))

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("Training vs Validation Loss")

plt.legend()

plt.show()