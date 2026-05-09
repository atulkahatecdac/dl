# -----------------------------------------------------
# Neural Network in Keras
# Loan Default Prediction
#
# Dataset:
# loan.csv
#
# Target:
# default
#
# Goal:
# Predict whether a customer
# will default on loan
# -----------------------------------------------------

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow import keras
from tensorflow.keras import layers


# -----------------------------------------------------
# STEP 1: Load Dataset
# -----------------------------------------------------

df = pd.read_csv("loan.csv")

print("\n--- First 5 Rows ---")
print(df.head())


# -----------------------------------------------------
# STEP 2: Handle Missing Values
# -----------------------------------------------------

df = df.dropna()


# -----------------------------------------------------
# STEP 3: Separate Features and Labels
# -----------------------------------------------------

X = df.drop("default", axis=1)

y = df["default"]


# -----------------------------------------------------
# STEP 4: Feature Scaling
# -----------------------------------------------------
#
# Neural networks train better
# when features are normalized
#

scaler = StandardScaler()

X = scaler.fit_transform(X)


# -----------------------------------------------------
# STEP 5: Train/Test Split
# -----------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42
)


# -----------------------------------------------------
# STEP 6: Build Neural Network
# -----------------------------------------------------
#
# Architecture:
#
# Input (8 features)
#        ↓
# Dense Layer (16 neurons)
#        ↓
# Dense Layer (8 neurons)
#        ↓
# Output Layer (1 neuron)
#

model = keras.Sequential([


    # Hidden Layer 1
    layers.Dense(

        16,

        activation='relu',

        input_shape=(8,)
    ),


    # Hidden Layer 2
    layers.Dense(

        8,

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
# STEP 8: Show Model Summary
# -----------------------------------------------------

print("\n--- Model Summary ---")

model.summary()


# -----------------------------------------------------
# STEP 9: Train Model
# -----------------------------------------------------

history = model.fit(

    X_train,
    y_train,

    epochs=50,

    batch_size=16,

    validation_split=0.2
)


# -----------------------------------------------------
# STEP 10: Evaluate Model
# -----------------------------------------------------

loss, accuracy = model.evaluate(

    X_test,
    y_test
)

print("\nTest Accuracy:", accuracy)


# -----------------------------------------------------
# STEP 11: Predict New Customer
# -----------------------------------------------------
#
# Features:
#
# age
# ed
# employ
# address
# income
# debtinc
# creddebt
# othdebt
#

new_customer = np.array([

    [35, 2, 10, 5, 50000, 15, 2000, 3000]
])


# Scale using SAME scaler
new_customer = scaler.transform(new_customer)


prediction = model.predict(new_customer)


print(

    "\nProbability of Default:",
    round(float(prediction[0][0]), 3)
)


# Convert probability to class
predicted_class = int(prediction[0][0] >= 0.5)

print(

    "Predicted Class:",
    predicted_class
)


# -----------------------------------------------------
# STEP 12: Simple Interpretation
# -----------------------------------------------------

if predicted_class == 1:

    print("\nCustomer likely to DEFAULT")

else:

    print("\nCustomer likely to NOT DEFAULT")