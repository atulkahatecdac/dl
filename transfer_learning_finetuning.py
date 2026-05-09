# -----------------------------------------------------
# Fine-Tuning using VGG16
# CIFAR-10 Image Classification
#
# Goal:
# Reuse pretrained VGG16 model
# AND retrain some deeper layers
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

X_train = tf.image.resize(X_train, (224, 224))
X_test = tf.image.resize(X_test, (224, 224))


# -----------------------------------------------------
# STEP 4: Load Pretrained VGG16
# -----------------------------------------------------
base_model = keras.applications.VGG16(

    weights='imagenet',

    include_top=False,

    input_shape=(224, 224, 3)
)


# -----------------------------------------------------
# STEP 5: Fine-Tuning Setup
# -----------------------------------------------------
#
# Freeze EARLY layers
# Unfreeze DEEPER layers
#
# Early layers:
# generic edges/textures
#
# Deeper layers:
# task-specific features
#

base_model.trainable = True


# Freeze all layers EXCEPT last 4
for layer in base_model.layers[:-4]:

    layer.trainable = False


# -----------------------------------------------------
# STEP 6: Build New Model
# -----------------------------------------------------
model = keras.Sequential([

    # Pretrained VGG16
    base_model,


    # Flatten feature maps
    layers.Flatten(),


    # Dense layer
    layers.Dense(

        256,

        activation='relu'
    ),

    layers.Dropout(0.5),


    # Output layer
    layers.Dense(

        10,

        activation='softmax'
    )
])


# -----------------------------------------------------
# STEP 7: Compile Model
# -----------------------------------------------------
#
# IMPORTANT:
# Use SMALL learning rate
# during fine-tuning
#
# Large learning rate can destroy
# pretrained weights
#

model.compile(

    optimizer=keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']
)


# -----------------------------------------------------
# STEP 8: Show Trainable Layers
# -----------------------------------------------------
print("\n--- Trainable Layers ---")

for layer in base_model.layers:

    print(
        layer.name,
        "→",
        layer.trainable
    )


# -----------------------------------------------------
# STEP 9: Model Summary
# -----------------------------------------------------
print("\n--- Fine-Tuning Model Summary ---")

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