# -----------------------------------------------------
# Transfer Learning using VGG16
# CIFAR-10 Image Classification
#
# Goal:
# Reuse pretrained VGG16 model
# instead of training CNN from scratch
# -----------------------------------------------------

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# -----------------------------------------------------
# STEP 1: Load CIFAR-10 Dataset
# -----------------------------------------------------
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()


# -----------------------------------------------------
# STEP 2: Normalize Images
# -----------------------------------------------------
X_train = X_train / 255.0
X_test = X_test / 255.0

# -----------------------------------------------------
# Reduce dataset size
# -----------------------------------------------------

# -----------------------------------------------------
# Reduce dataset size
# -----------------------------------------------------

X_train = X_train[:1000]
y_train = y_train[:1000]

X_test = X_test[:200]
y_test = y_test[:200]

# -----------------------------------------------------
# STEP 3: Resize Images
# -----------------------------------------------------
# VGG16 expects:
# 224 x 224 x 3
#
# CIFAR-10 images are:
# 32 x 32 x 3
#
# So we resize them.

X_train = tf.image.resize(X_train, (224, 224))
X_test = tf.image.resize(X_test, (224, 224))


# -----------------------------------------------------
# STEP 4: Load Pretrained VGG16
# -----------------------------------------------------
#
# weights='imagenet'
# -> load pretrained weights
#
# include_top=False
# -> remove original classifier
#

base_model = keras.applications.VGG16(

    weights='imagenet',

    include_top=False,

    input_shape=(224, 224, 3)
)


# -----------------------------------------------------
# STEP 5: Freeze Pretrained Layers
# -----------------------------------------------------
#
# Do NOT retrain VGG16 weights
#

base_model.trainable = False


# -----------------------------------------------------
# STEP 6: Build New Model
# -----------------------------------------------------
#
# Add our own classifier layers
#

model = keras.Sequential([

    # Pretrained VGG16 feature extractor
    base_model,


    # Convert feature maps to vector
    layers.Flatten(),


    # Dense layer
    layers.Dense(

        256,

        activation='relu'
    ),

    layers.Dropout(0.5),


    # Output layer
    # CIFAR-10 = 10 classes

    layers.Dense(

        10,

        activation='softmax'
    )
])


# -----------------------------------------------------
# STEP 7: Compile Model
# -----------------------------------------------------
model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']
)


# -----------------------------------------------------
# STEP 8: Show Model Summary
# -----------------------------------------------------
print("\n--- Transfer Learning Model Summary ---")

model.summary()


# -----------------------------------------------------
# STEP 9: Train Model
# -----------------------------------------------------
history = model.fit(

    X_train,
    y_train,

    epochs=2,

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
# STEP 11: Predict One Image
# -----------------------------------------------------
predictions = model.predict(X_test)

predicted_class = predictions[0].argmax()

print("\nPredicted Class:", predicted_class)

print("Actual Class   :", y_test[0][0])


# -----------------------------------------------------
# STEP 12: CIFAR-10 Labels
# -----------------------------------------------------
class_names = [

    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

print(
    "\nPredicted Label:",
    class_names[predicted_class]
)