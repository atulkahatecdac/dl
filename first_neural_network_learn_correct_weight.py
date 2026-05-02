# -----------------------------------------------------
# Simple Linear Model: y = w * x
# Goal: learn correct weight (should be ~2)
# -----------------------------------------------------

inputs = [1, 2, 3, 4]
targets = [2, 4, 6, 8]

w = 0.1
learning_rate = 0.1


def predict(i):
    return w * i

costs = []

# -----------------------------------------------------
# TRAINING LOOP
# -----------------------------------------------------
for step in range(25):

    # Step 1: Forward pass (predictions)
    pred = [predict(i) for i in inputs]

    # Step 2: Compute squared error (loss per sample)
    errors = [(p - t) ** 2 for p, t in zip(pred, targets)]

    # Step 3: Compute cost (average loss)
    cost = sum(errors) / len(targets)
    costs.append(cost)

    print(f"Step {step+1} | Weight: {w:.4f}, Cost: {cost:.4f}")

    # -------------------------------------------------
    # BACKPROP (this is the key learning part)
    # -------------------------------------------------

    # Step 4: Derivative of loss w.r.t prediction
    # d/dŷ (y - ŷ)^2 = 2(ŷ - y)
    # This tells: "how wrong is the prediction?"
    errors_d = [2 * (p - t) for p, t in zip(pred, targets)]

    # Step 5: Convert this into derivative w.r.t weight
    # Since ŷ = w * x, change in ŷ depends on x
    # So multiply by input (chain rule)
    # weight_d = error signal × input
    weight_d = [e * x for e, x in zip(errors_d, inputs)]

    # Step 6: Average gradient across all samples (batch gradient)
    gradient = sum(weight_d) / len(weight_d)

    # -------------------------------------------------
    # Step 7: Update weight (gradient descent)
    # w = w - learning_rate × gradient
    #
    # NOTE:
    # - If gradient is negative → w increases
    # - If gradient is positive → w decreases
    # -------------------------------------------------
    w = w - learning_rate * gradient


# -----------------------------------------------------
# TEST THE MODEL
# -----------------------------------------------------
print("\n--- Testing ---")

test_inputs = [5, 6]
test_targets = [10, 12]

pred = [predict(i) for i in test_inputs]

for i, t, p in zip(test_inputs, test_targets, pred):
    print(f"input:{i}, target:{t}, pred:{p:.4f}")

# Plot costs 
import matplotlib.pyplot as plt

plt.figure()

# Plot cost vs iteration
plt.plot(range(1, len(costs) + 1), costs, marker='o')

plt.title("Cost vs Iteration")
plt.xlabel("Iteration")
plt.ylabel("Cost")

plt.grid(True)

plt.show()