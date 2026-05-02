inputs = [1, 2, 3, 4]
targets = [2, 4, 6, 8]

# Slope of the line - Actually the computer needs to calculate this, but we will start with an arbitrary value
w = 0.1
learning_rate = 0.1

def predict(i):
    return w * i

# train the network
costs = []
for i in range(25):
    pred = [predict(i) for i in inputs]
    errors = [t - p for t, p in zip(targets, pred)]
    cost = sum(errors) / len(targets)
    costs.append(cost)

    # Note we are calling slope as weight, since in our example the slope is the weight
    # print(f"Weight: {w}, Cost: {cost}")
    # Print weight and cost up to two decimal places
    print(f"Weight: {w:.4f}, Cost: {cost:.4f}")
    w += learning_rate * cost

# test the network
test_inputs = [5, 6, 7, 8]
test_targets = [10, 12, 14, 16]
pred = [predict(i) for i in test_inputs]
for i, t, p in zip(test_inputs, test_targets, pred):
    print(f"Input: {i}, Target: {t}, Prediction: {p:.4f}")
    
# Print the weight
print(f"Weight: {w:.2f}")