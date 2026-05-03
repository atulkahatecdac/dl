import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# 1. Training Data
x = np.array([1, 2, 3, 4])
y = np.array([2, 4, 6, 8])

# 2. Slopes (weights) to animate through
weights = np.linspace(0.5, 3.5, 40)

# 3. Setup the plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x, y, color='red', s=100, label='Target Data')
line, = ax.plot(x, weights[0] * x, color='blue', label='Predicted Line (w*x)')

ax.set_xlim(0, 5)
ax.set_ylim(0, 15)
ax.set_xlabel('Inputs (x)')
ax.set_ylabel('Targets (y)')
ax.legend()

# 4. Animation Function
def update(frame):
    w = weights[frame]
    line.set_ydata(w * x) # Update the line's y-values based on the new slope
    ax.set_title(f"Current Slope (w): {w:.2f}", fontsize=14)
    return line,

# 5. Run and show animation
ani = animation.FuncAnimation(fig, update, frames=len(weights), interval=100)
plt.show()