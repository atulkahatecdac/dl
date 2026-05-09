# -----------------------------------------------------
# Transfer Learning using MobileNetV2
# CIFAR-10 Image Classification
#
# Goal:
# Faster and lighter transfer learning
# compared to VGG16
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
# STEP 3: Reduce Dataset Size
# -----------------------------------------------------
#
# Small subset for fast classroom demo
#

X_train = X_train[:1000]
y_train = y_train[:1000]

X_test = X_test[:200]
y_test = y_test[:200]


# -----------------------------------------------------
# STEP 4: Resize Images
# -----------------------------------------------------
#
# MobileNetV2 expects larger images
#

X_train = tf.image.resize(X_train, (96, 96))
X_test = tf.image.resize(X_test, (96, 96))


# -----------------------------------------------------
# STEP 5: Load Pretrained MobileNetV2
# -----------------------------------------------------
#
# weights='imagenet'
# -> pretrained ImageNet weights
#
# include_top=False
# -> remove original classifier
#

base_model = keras.applications.MobileNetV2(

    weights='imagenet',

    include_top=False,

    input_shape=(96, 96, 3)
)


# -----------------------------------------------------
# STEP 6: Freeze Pretrained Layers
# -----------------------------------------------------
#
# Feature Extraction
#

base_model.trainable = False


# -----------------------------------------------------
# STEP 7: Build New Model
# -----------------------------------------------------
#
# GlobalAveragePooling2D
# is much more efficient than Flatten
#

model = keras.Sequential([

    # Pretrained feature extractor
    base_model,


    # Compress feature maps
    layers.GlobalAveragePooling2D(),


    # Dense layer
    layers.Dense(

        128,

        activation='relu'
    ),

    layers.Dropout(0.3),


    # Output layer
    layers.Dense(

        10,

        activation='softmax'
    )
])


# -----------------------------------------------------
# STEP 8: Compile Model
# -----------------------------------------------------
model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']
)


# -----------------------------------------------------
# STEP 9: Show Model Summary
# -----------------------------------------------------
print("\n--- MobileNetV2 Transfer Learning Model ---")

model.summary()


# -----------------------------------------------------
# STEP 10: Train Model
# -----------------------------------------------------
history = model.fit(

    X_train,
    y_train,

    epochs=2,

    batch_size=16,

    validation_split=0.2
)


# -----------------------------------------------------
# STEP 11: Evaluate Model
# -----------------------------------------------------
loss, accuracy = model.evaluate(

    X_test,
    y_test
)

print("\nTest Accuracy:", accuracy)


# -----------------------------------------------------
# STEP 12: Predict One Image
# -----------------------------------------------------
predictions = model.predict(X_test)

predicted_class = predictions[0].argmax()

print("\nPredicted Class:", predicted_class)

print("Actual Class   :", y_test[0][0])


# -----------------------------------------------------
# STEP 13: CIFAR-10 Labels
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