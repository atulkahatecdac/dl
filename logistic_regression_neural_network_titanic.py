# -----------------------------------------------------
# Logistic Regression using neural network (Titanic)
# Features: pclass + gender
# -----------------------------------------------------

import pandas as pd
import math

# -----------------------------------------------------
# Load data
# -----------------------------------------------------
df = pd.read_csv("titanic.csv")

# Keep only relevant columns
df = df[["survived", "pclass", "gender"]].dropna()

# Encode gender: male=0, female=1
df["gender"] = df["gender"].map({"male": 0, "female": 1})

# -----------------------------------------------------
# Features and target
# -----------------------------------------------------
X = df[["pclass", "gender"]].values.tolist()
y = df["survived"].tolist()

# -----------------------------------------------------
# Simple scaling (important!)
# pclass is 1,2,3 → scale to 0-1
# -----------------------------------------------------
for row in X:
    row[0] = row[0] / 3   # scale pclass

# -----------------------------------------------------
# Initialize weights
# -----------------------------------------------------
w = [0.1, 0.1]   # one weight per feature
b = 0.1

learning_rate = 0.4
epochs = 1000


# -----------------------------------------------------
# Sigmoid activation
# -----------------------------------------------------
def sigmoid(z):
    return 1 / (1 + math.exp(-z))


# -----------------------------------------------------
# Prediction
# -----------------------------------------------------
def predict(x):
    z = sum(w[i] * x[i] for i in range(len(x))) + b
    return sigmoid(z)


# -----------------------------------------------------
# Training loop
# -----------------------------------------------------
for step in range(epochs):

    preds = [predict(xi) for xi in X]

    # Binary cross-entropy loss
    loss = 0
    for p, t in zip(preds, y):
        loss += -(t * math.log(p + 1e-9) + (1 - t) * math.log(1 - p + 1e-9))
    loss /= len(y)

    if step % 10 == 0:
        print(f"Step {step} | Loss: {loss:.4f}")

    # -------------------------------------------------
    # Gradient (very simple!)
    # derivative = (pred - target)
    # -------------------------------------------------
    errors = [(p - t) for p, t in zip(preds, y)]

    # Gradient for weights
    grad_w = [0, 0]
    for i in range(len(w)):
        grad_w[i] = sum(errors[j] * X[j][i] for j in range(len(X))) / len(X)

    # Gradient for bias
    grad_b = sum(errors) / len(errors)

    # -------------------------------------------------
    # Update
    # -------------------------------------------------
    for i in range(len(w)):
        w[i] -= learning_rate * grad_w[i]

    b -= learning_rate * grad_b


# -----------------------------------------------------
# Testing (manual examples)
# -----------------------------------------------------
print("\n--- Testing ---")

# Example: 1st class female
test = [1/3, 1]   # pclass=1, female=1
p = predict(test)
print("1st class female →", round(p, 3), "→", 1 if p >= 0.5 else 0)

# Example: 3rd class male
test = [3/3, 0]   # pclass=3, male=0
p = predict(test)
print("3rd class male →", round(p, 3), "→", 1 if p >= 0.5 else 0)