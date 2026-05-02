inputs = [1, 2, 3, 4]
targets = [2, 4, 6, 8]

# Slope of the line - Actually the computer needs to calculate this, but we will start with an arbitrary value
w = 0.1
learning_rate = 0.01

def predict(i):
    return w * i

# train the network
costs = []
for i in range(10):
    pred = [predict(i) for i in inputs]
    errors = [t - p for t, p in zip(targets, pred)]
    cost = sum(errors) / len(targets)
    costs.append(cost)

    # Note we are calling slope as weight, since in our example the slope is the weight
    print(f"Weight: {w}, Cost: {cost}")
    w += learning_rate * cost

# Plot the cost

import matplotlib.pyplot as plt

plt.plot(range(10), costs, marker='o')
plt.title("Cost over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.grid(True)
plt.show() # May not work in VS-Code, so save the plot
plt.savefig("cost_plot.png")