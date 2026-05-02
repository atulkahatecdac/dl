# -----------------------------------------------------
# Neural Network (Keras) - Titanic Survival Prediction
# Features used: pclass + gender
#
# Goal:
# Learn to predict whether a passenger survived (0/1)
# -----------------------------------------------------

import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers


# -----------------------------------------------------
# STEP 1: Load dataset
# -----------------------------------------------------
df = pd.read_csv("titanic.csv")

# Keep only relevant columns
# survived = target (0 or 1)
# pclass   = passenger class (1,2,3)
# gender   = categorical feature
df = df[["survived", "pclass", "gender"]].dropna()

# -----------------------------------------------------
# STEP 2: Convert categorical data → numeric
# -----------------------------------------------------
# Neural networks cannot understand text like "male/female"
# So we convert:
# male → 0
# female → 1
df["gender"] = df["gender"].map({"male": 0, "female": 1})


# -----------------------------------------------------
# STEP 3: Prepare features (X) and labels (y)
# -----------------------------------------------------
X = df[["pclass", "gender"]].values   # input features
y = df["survived"].values            # output (target)


# -----------------------------------------------------
# STEP 4: Feature scaling
# -----------------------------------------------------
# pclass values are 1,2,3 → scale to 0–1 range
# Helps training become stable and faster
X[:, 0] = X[:, 0] / 3


# -----------------------------------------------------
# STEP 5: Build the neural network
# -----------------------------------------------------
# Structure:
# Input (2 features) → Hidden layer → Output

model = keras.Sequential([
    
    # Hidden layer with 4 neurons
    # ReLU helps model learn non-linear patterns
    layers.Dense(4, activation='relu', input_shape=(2,)),
    
    # Output layer:
    # 1 neuron + sigmoid → gives probability (0 to 1)
    layers.Dense(1, activation='sigmoid')
])


# -----------------------------------------------------
# STEP 6: Compile the model
# -----------------------------------------------------
# optimizer: how weights are updated (Adam = smart gradient descent)
# loss: binary classification → binary_crossentropy
# metrics: track accuracy
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# -----------------------------------------------------
# STEP 7: Train the model
# -----------------------------------------------------
# epochs = number of passes over entire dataset
# batch_size = how many samples per update
model.fit(X, y, epochs=50, batch_size=16)


# -----------------------------------------------------
# STEP 8: Testing / Predictions
# -----------------------------------------------------
print("\n--- Testing ---")

# Example 1: 1st class female
# pclass = 1 → scaled = 1/3
# gender = 1 (female)
import numpy as np

test = np.array([[1/3, 1]])
p = model.predict(test)

print("1st class female →", round(float(p[0][0]), 3),
      "→", int(p[0][0] >= 0.5))


# Example 2: 3rd class male
# pclass = 3 → scaled = 3/3 = 1
# gender = 0 (male)
test = np.array([[1, 0]])
p = model.predict(test)

print("3rd class male →", round(float(p[0][0]), 3),
      "→", int(p[0][0] >= 0.5))