import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# -----------------------------------------------------
# STEP 0: Create simple training data
# -----------------------------------------------------
# Feature 1: Hours studied (0 to 10)
# Feature 2: Hours slept (0 to 10)
# The Rule: If (Studied + Slept) > 10, the student passes (1). Otherwise, they fail (0).

# Generate 1000 random student records
X_train = np.random.uniform(0, 10, (1000, 2))
# Calculate the pass/fail label based on our rule
y_train = (X_train[:, 0] + X_train[:, 1] > 10).astype(int)


# -----------------------------------------------------
# STEP 1: Start the empty conveyor belt
# -----------------------------------------------------
model = keras.Sequential()


# -----------------------------------------------------
# STEP 2: Add your processing nodes (Layers)
# -----------------------------------------------------
# IMPORTANT CHANGE: input_shape is now (2,) because we only have 2 features (Study & Sleep)
model.add(layers.Dense(8, activation='relu', input_shape=(2,)))

# Output layer (predicting a single yes/no value)
model.add(layers.Dense(1, activation='sigmoid'))


# -----------------------------------------------------
# STEP 3: Hire the manager (Compile)
# -----------------------------------------------------
# Added 'metrics=['accuracy']' so we can watch it get smarter during training
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# -----------------------------------------------------
# STEP 4: Turn on the machine (Fit)
# -----------------------------------------------------
print("Training the model...")
model.fit(X_train, y_train, epochs=15, verbose=1)


# -----------------------------------------------------
# STEP 5: Test it out!
# -----------------------------------------------------
print("\n--- Testing New Students ---")

# Let's test two new students the model has never seen:
# Student A: 8 hours studying, 7 hours sleep (Total 15 -> Should easily pass)
# Student B: 2 hours studying, 4 hours sleep (Total 6 -> Should fail)
test_students = np.array([[8, 7], 
                          [2, 4]])

predictions = model.predict(test_students)

print(f"Student A (8h study, 7h sleep) pass probability: {predictions[0][0]:.2f}")
print(f"Student B (2h study, 4h sleep) pass probability: {predictions[1][0]:.2f}")