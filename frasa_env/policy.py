import matplotlib.pyplot as plt
import numpy as np

# Data for specialized vs. shared policy success rates
robots = ["Bez1", "OP3 Rot", "Bez2", "Bez3", "Sig", "BitBot", "Nugus"]
specialized = np.array([0.38, 0.93, 0.98, 0.95, 1.00, 0.92, 0.00])
shared_best = np.array([0.92, 0.87, 0.97, 0.83, 1.00, 0.92, 0.89])

# Compute delta (shared - specialized)
delta = shared_best - specialized

x = np.arange(len(robots))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))

# Plot grouped bars
bar1 = ax.bar(x - width/2, specialized, width, label='Specialized Policy', color='#FFC107')
bar2 = ax.bar(x + width/2, shared_best, width, label='Best Shared Policy', color='#FF5722')

# Plot delta as a dashed line with markers
ax.plot(x, delta, marker='x', color='black', linestyle='--', linewidth=2, label='Δ (Shared - Spec)', zorder=5)

# Labels and title
ax.set_xticks(x)
ax.set_xticklabels(robots, rotation=45, ha="right")
ax.set_ylabel('Success Rate')
ax.set_title('Specialized vs. Best Shared Policy Performance')
ax.set_ylim(-0.2, 1.35)

# Annotate bar heights
for bar in bar1 + bar2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{height:.2f}", ha='center', va='bottom')

# Annotate delta points
for xi, d in zip(x, delta):
    va = 'bottom' if d >= 0 else 'top'
    offset = 0.02 if d >= 0 else -0.02
    offset2 = 0.3 if d == 0.89 else 0
    offset3 = -0.05 if d == 0.89 else 0
    ax.text(xi-offset2, d + offset + offset3, f"{d:+.2f}", ha='center', va=va)

# Legend and grid
ax.legend(loc='upper left')
ax.grid(axis='y', linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()
