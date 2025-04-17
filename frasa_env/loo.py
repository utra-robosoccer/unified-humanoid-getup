import matplotlib.pyplot as plt
import numpy as np

# Data from the LOO experiment
robots = ["Bez1", "OP3 Rot", "Bez2", "Bez3", "Sig", "BitBot", "Nugus"]
data = np.array([
    [0.58, 0.62, 0.84, 0.74, 0.96, 0.93, 0.61],
    [0.02, 0.08, 0.88, 0.90, 0.93, 0.91, 0.88],
    [0.00, 0.87, 0.51, 0.93, 0.96, 0.92, 0.62],
    [0.36, 0.89, 0.89, 0.05, 1.00, 0.93, 0.82],
    [0.62, 0.78, 0.92, 0.95, 0.71, 0.93, 0.88],
    [0.92, 0.87, 0.97, 0.83, 1.00, 0.92, 0.89],
    [0.34, 0.61, 0.77, 0.61, 0.97, 0.89, 0.22]
])

fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.imshow(data, cmap='viridis', vmin=0, vmax=1)
ax.xaxis.tick_top()             # move x‑axis ticks to the top
ax.xaxis.set_label_position('top')  # move the x‑axis label as well
# Set ticks and labels
ax.set_xticks(np.arange(len(robots)))
ax.set_yticks(np.arange(len(robots)))
ax.set_xticklabels(robots)
ax.set_yticklabels(robots)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

# Annotate cells with values
for i in range(len(robots)):
    for j in range(len(robots)):
        ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                color="w" if data[i, j] < 0.5 else "black")

# Labels and colorbar
ax.set_xlabel("Tested on")
ax.set_ylabel("Held out")
ax.set_title("LOO Zero-Shot Get-Up Success Heatmap")
fig.colorbar(cax, label="Success Rate")
plt.tight_layout()
plt.show()
