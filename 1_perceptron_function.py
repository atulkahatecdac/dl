# Perceptron function

# First element in vector x must be 1 (bias input)
# Length of w and x must be n+1 for neuron with n inputs.

def compute_output(w, x):
    z = 0.0
    
    for i in range(len(w)):
        z += x[i] * w[i]   # Compute weighted sum
    
    # Apply sign activation function
    if z < 0:
        return -1
    else:
        return 1


# --- TEST CASES ---
print(compute_output([0.9, -0.6, -0.5], [1.0, -1.0, -1.0]))
print(compute_output([0.9, -0.6, -0.5], [1.0, -1.0, 1.0]))
print(compute_output([0.9, -0.6, -0.5], [1.0, 1.0, -1.0]))
print(compute_output([0.9, -0.6, -0.5], [1.0, 1.0, 1.0]))