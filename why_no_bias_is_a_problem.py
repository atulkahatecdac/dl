import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------------------------------
# 1. Setup the Data
# -----------------------------------------------------
# Our target rule is y = 2x + 10
x_data = np.array([1, 2, 3, 4])
y_data = np.array([12, 14, 16, 18]) 

# -----------------------------------------------------
# 2. Setup the Animation Frames
# -----------------------------------------------------
num_frames = 80
weights = np.zeros(num_frames)
biases = np.zeros(num_frames)

# Phase 1 (Frames 0 to 40): Keep weight=1.0, animate bias from 0 to 10
weights[:40] = 1.0
biases[:40] = np.linspace(0, 10, 40)

# Phase 2 (Frames 40 to 80): Keep bias=10.0, animate weight from 1.0 to 2.0
weights[40:] = np.linspace(1.0, 2.0, 40)
biases[40:] = 10.0

# -----------------------------------------------------
# 3. Setup the Plot
# -----------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 5)
ax.set_ylim(0, 25)
ax.set_xlabel("Input (x)", fontsize=12)
ax.set_ylabel("Prediction (y)", fontsize=12)
ax.set_title("The Missing Piece: Why We Need Bias", fontsize=14)
ax.grid(True, linestyle='--', alpha=0.6)

# Plot the target data points in red
ax.plot(x_data, y_data, 'ro', markersize=10, label='Target Data')

# Initialize the moving prediction line in blue
line, = ax.plot([], [], 'b-', linewidth=2.5, label='Prediction Line')

# Add a text box to show the live equation and current phase
info_text = ax.text(0.05, 0.80, '', transform=ax.transAxes, fontsize=14,
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

ax.legend(loc="lower right")

# -----------------------------------------------------
# 4. The Animation Engine
# -----------------------------------------------------
def init():
    line.set_data([], [])
    info_text.set_text('')
    return line, info_text

def animate(i):
    w = weights[i]
    b = biases[i]
    
    # Draw the line straight across our X-axis bounds
    x_line = np.linspace(0, 5, 100)
    y_line = w * x_line + b
    line.set_data(x_line, y_line)
    
    # Update the text box to explain what is happening
    if i < 40:
        action = "PHASE 1: Lifting the line (Increasing Bias)"
    else:
        action = "PHASE 2: Tilting the line (Increasing Weight)"
        
    info_text.set_text(f"{action}\nEquation: Prediction = {w:.2f}*x + {b:.2f}")
    
    return line, info_text

# -----------------------------------------------------
# 5. Run it!
# -----------------------------------------------------
# interval=80 means 80 milliseconds per frame
ani = FuncAnimation(fig, animate, init_func=init, frames=num_frames, 
                    interval=80, blit=True, repeat_delay=3000)

plt.show()