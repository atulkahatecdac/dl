# -----------------------------------------------------
# Basic CNN Example - MNIST Digit Classification
#
# Goal:
# Recognize handwritten digits (0-9)
#
# Concepts:
#   1. Convolution Layer
#   2. ReLU Activation
#   3. MaxPooling
#   4. Flatten
#   5. Dense Layer
#   6. Softmax Output
# -----------------------------------------------------

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# -----------------------------------------------------
# STEP 1: Load Dataset
# -----------------------------------------------------
# MNIST:
# 28x28 grayscale handwritten digit images

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()


# -----------------------------------------------------
# STEP 2: Normalize Images
# -----------------------------------------------------
# Convert pixel values:
# 0-255  →  0-1

X_train = X_train / 255.0
X_test = X_test / 255.0


# -------------------------------------------------------------------------------------
# STEP 3: Reshape Images into CNN format
# Original: (60000, 28, 28), meaning 60000 images, each of size 28x28
# CNN expected format: (samples, height, width, channels) 
# Channel = Colours = 1 for grayscale (only one intensity value, 0 = black, 255 = white)
# (-1, 28, 28, 1) means (automatically calculate number of images, 28x28, 1 channel)
# We could have also written this as: (60000, 28, 28, 1)
# -------------------------------------------------------------------------------------
# CNN expects:
#
# (height, width, channels)
#
# MNIST is grayscale:
# channels = 1

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)


# -----------------------------------------------------
# STEP 4: Build CNN
# -----------------------------------------------------
#
# Architecture:
#
# Input Image
#      ↓
# Conv2D
#      ↓
# ReLU
#      ↓
# MaxPooling
#      ↓
# Flatten
#      ↓
# Dense
#      ↓
# Softmax Output
#

model = keras.Sequential([

    # -------------------------------------------------
    # Convolution Layer
    # -------------------------------------------------
    # 32 filters
    # 3x3 kernel
    #
    # Learns:
    # edges, curves, patterns

    layers.Conv2D(  # Means scan image using small filters

        filters=32, #  Learn 32 different visual patterns (e.g. filter 1: vertical edges, filter 2: horizontal edges, etc)

        kernel_size=(3, 3), # Look at tiny 3x3 neighborhood

        activation='relu', # Keep only positive values (useful) and kill negative values (not of use)

        input_shape=(28, 28, 1)
    ),


    # -------------------------------------------------
    # Pooling Layer
    # -------------------------------------------------
    # Reduces image size
    # Keeps important features

    layers.MaxPooling2D(  # Reduce image size / Shrink feature maps

        pool_size=(2, 2)
    ),


    # -------------------------------------------------
    # Flatten Layer
    # -------------------------------------------------
    # Convert 2D feature maps
    # into 1D vector

    layers.Flatten(), # Input: (13, 13, 32)  Output: (5408,)


    # -------------------------------------------------
    # Dense Layer
    # -------------------------------------------------
    layers.Dense( # Combine learned visual features

        64, # Learn 64 different visual patterns

        activation='relu'
    ),


    # -------------------------------------------------
    # Output Layer
    # -------------------------------------------------
    # 10 neurons:
    # digits 0-9

    layers.Dense( # Make prediction

        10, # 10 neurons, one for each digit (0-9)

        activation='softmax' # Choose the one with the highest probability
    )
])


# -----------------------------------------------------
# STEP 5: Compile Model
# -----------------------------------------------------
model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']
)


# -----------------------------------------------------
# STEP 6: Train Model
# -----------------------------------------------------
model.fit(

    X_train,
    y_train,

    epochs=5,

    batch_size=32
)


# -----------------------------------------------------
# STEP 7: Evaluate Model
# -----------------------------------------------------
loss, accuracy = model.evaluate(

    X_test,
    y_test
)

print("\nTest Accuracy:", accuracy)


# -----------------------------------------------------
# STEP 8: Make Prediction
# -----------------------------------------------------
predictions = model.predict(X_test)

# Get predicted digit
predicted_digit = predictions[0].argmax()

print("\nPredicted Digit:", predicted_digit)

print("Actual Digit   :", y_test[0])