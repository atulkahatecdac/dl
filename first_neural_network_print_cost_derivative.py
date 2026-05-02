inputs = [1, 2, 3, 4]
targets = [2, 4, 6, 8]

w = 0.1
learning_rate = 0.1

def predict(i):
    return w * i

# train the network
for _ in range(25):
    pred = [predict(i) for i in inputs]
    errors = [(p - t) ** 2 for p, t in zip(pred, targets)]
    cost = sum(errors) / len(targets)

    print(f"Weight: {w:.2f}, Cost: {cost:.2f}")

    # derivative of MSE w.r.t prediction
    errors_d = [2 * (p - t) for p, t in zip(pred, targets)]
	# weight_d = derivative of cost w.r.t weight (for each data point)
    weight_d = [e * i for e, i in zip(errors_d, inputs)]
    print(weight_d)

    exit()  # stop after first iteration for inspection

    w += learning_rate * cost


# test the network
test_inputs = [5, 6]
test_targets = [10, 12]

pred = [predict(i) for i in test_inputs]

for i, t, p in zip(test_inputs, test_targets, pred):
    print(f"input:{i}, target:{t}, pred:{p:.4f}")