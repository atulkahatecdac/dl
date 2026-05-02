# -----------------------------------------------------
# Linear model with bias: y = w*x + b
# Goal: learn both slope (w) and intercept (b)
# -----------------------------------------------------

inputs = [1, 2, 3, 4]
targets = [12, 14, 16, 18]   # roughly: y = 2x + 10

w = 0.1   # weight (slope)
b = 0.3   # bias (intercept)

learning_rate = 0.1
epochs = 40


def predict(i):
    return w * i + b


# -----------------------------------------------------
# TRAINING LOOP
# -----------------------------------------------------
for step in range(epochs):

    # Step 1: Forward pass (predictions)
    pred = [predict(i) for i in inputs]

    # Step 2: Compute squared error for each sample
    errors = [(p - t) ** 2 for p, t in zip(pred, targets)]

    # Step 3: Compute cost (Mean Squared Error)
    cost = sum(errors) / len(targets)

    print(f"Step {step+1:2d} | Weight: {w:.4f}, Bias: {b:.4f}, Cost: {cost:.4f}")

    # -------------------------------------------------
    # BACKPROP (Gradient computation)
    # -------------------------------------------------

    # Derivative of loss w.r.t prediction
    # d/dŷ (y - ŷ)^2 = 2(ŷ - y)
    errors_d = [2 * (p - t) for p, t in zip(pred, targets)]

    # -------------------------------------------------
    # Gradient w.r.t weight (w)
    # Since ŷ = w*x + b, change in ŷ depends on x
    # So multiply error signal by input
    # weight_d = error × input
    # -------------------------------------------------
    weight_d = [e * x for e, x in zip(errors_d, inputs)]

    # -------------------------------------------------
    # Gradient w.r.t bias (b)
    # Since ŷ = w*x + b, change in ŷ w.r.t b is 1
    # So multiply error signal by 1
    # bias_d = error × 1
    # -------------------------------------------------
    bias_d = [e * 1 for e in errors_d]

    # -------------------------------------------------
    # Average gradients (batch gradient descent)
    # -------------------------------------------------
    grad_w = sum(weight_d) / len(weight_d)
    grad_b = sum(bias_d) / len(bias_d)

    # -------------------------------------------------
    # Update parameters
    # w = w - lr * gradient
    # b = b - lr * gradient
    # -------------------------------------------------
    w = w - learning_rate * grad_w
    b = b - learning_rate * grad_b


# -----------------------------------------------------
# TEST THE MODEL
# -----------------------------------------------------
print("\n--- Testing ---")

test_inputs = [5, 6]
test_targets = [20, 22]

pred = [predict(i) for i in test_inputs]

for i, t, p in zip(test_inputs, test_targets, pred):
    print(f"input:{i}, target:{t}, pred:{p:.4f}")