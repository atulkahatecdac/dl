import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------------------------------
# 1. Define the Math
# -----------------------------------------------------
def cost_function(weight):
    return (weight - 3)**2 + 1

def gradient(weight):
    return 2 * (weight - 3)

# -----------------------------------------------------
# 2. Setup Gradient Descent Parameters
# -----------------------------------------------------
learning_rate = 0.1
initial_weight = 0.1  # for negative slope
# initial_weight = 5.9  # for positive slope
num_steps = 40

weights_history = [initial_weight]
for _ in range(num_steps):
    current_weight = weights_history[-1]
    grad = gradient(current_weight)
    
    next_weight = current_weight - (learning_rate * grad)
    weights_history.append(next_weight)

# -----------------------------------------------------
# 3. Setup the Plot
# -----------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-1, 7)
ax.set_ylim(0, 15)
ax.set_xlabel("Weight", fontsize=12)
ax.set_ylabel("Cost", fontsize=12)
ax.set_title("Gradient Descent: Following the Slope to the Minimum", fontsize=14)
ax.grid(True, linestyle='--', alpha=0.6)

# Draw the static U-shaped curve
x_vals = np.linspace(-1, 7, 100)
y_vals = cost_function(x_vals)
ax.plot(x_vals, y_vals, color='#2c3e50', linewidth=2, label='Cost Function')

# Initialize the elements that will move
point, = ax.plot([], [], 'ro', markersize=12, label='Current State')
path_line, = ax.plot([], [], 'r--', alpha=0.5)

# NEW: Initialize the Tangent (Slope) line
tangent_line, = ax.plot([], [], 'g-', linewidth=2.5, label='Tangent Line (Slope)')

info_text = ax.text(0.05, 0.80, '', transform=ax.transAxes, fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

ax.legend()

# -----------------------------------------------------
# 4. The Animation Engine
# -----------------------------------------------------
def init():
    point.set_data([], [])
    path_line.set_data([], [])
    tangent_line.set_data([], []) # Clear tangent
    info_text.set_text('')
    return point, path_line, tangent_line, info_text

def animate(i):
    current_w = weights_history[i]
    current_cost = cost_function(current_w)
    current_grad = gradient(current_w)
    
    # Update point and path
    point.set_data([current_w], [current_cost])
    path_line.set_data(weights_history[:i+1], [cost_function(w) for w in weights_history[:i+1]])
    
    # Calculate and draw the tangent line
    # We create a short line segment extending 1.5 units left and right of the current point
    x_tangent = np.array([current_w - 1.5, current_w + 1.5])
    # Equation of a line: y = m(x - x1) + y1
    y_tangent = current_grad * (x_tangent - current_w) + current_cost
    tangent_line.set_data(x_tangent, y_tangent)
    
    text_content = (
        f"Step: {i}/{num_steps}\n"
        f"Weight: {current_w:.3f}\n"
        f"Cost: {current_cost:.3f}\n"
        f"Slope (Gradient): {current_grad:.2f}"
    )
    info_text.set_text(text_content)
    
    return point, path_line, tangent_line, info_text

# -----------------------------------------------------
# 5. Run it!
# -----------------------------------------------------
ani = FuncAnimation(fig, animate, init_func=init, frames=len(weights_history), 
                    interval=200, blit=True, repeat_delay=2000)

plt.show()