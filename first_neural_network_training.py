# Training data
inputs = [1, 2, 3, 4]
targets = [2, 4, 6, 8]

# Slope of the line - Actually the computer needs to calculate this, but we will start with an arbitrary value
w = 0.1 

def predict(i):
    return w * i

# train the network
pred = [predict(i) for i in inputs] # List of predicted values
errors = [t - p for t, p in zip(targets, pred)] # Difference betweeen actual target and predicted value
cost = sum(errors) / len(targets) # Average of all errors - A single number that tells us how well our network is doing

# Note we are calling slope as weight, since in our example the slope is the weight
print(f"Weight: {w}, Cost: {cost}")