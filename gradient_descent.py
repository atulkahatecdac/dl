import matplotlib.pyplot as plt

# Dataset
inputs = [1, 2, 3, 4]
targets = [2, 4, 6, 8]

# Try a range of weights from -1 to 5
weights = [w / 100 for w in range(-100, 500)]  # from -1.0 to 5.0
costs = []

# Mean Squared Error for each weight
for w in weights:
    predictions = [w * x for x in inputs]
    errors = [(t - p) ** 2 for t, p in zip(targets, predictions)]
    cost = sum(errors) / len(errors)
    costs.append(cost)

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(weights, costs, color='blue')
plt.title("Cost vs Weight (Gradient Descent Curve)")
plt.xlabel("Weight (w)")
plt.ylabel("Cost (MSE)")
plt.grid(True)
plt.axvline(x=2.0, color='red', linestyle='--', label='True weight = 2')
plt.legend()
plt.tight_layout()
plt.savefig("gradient_descent_curve.png")
plt.show()
