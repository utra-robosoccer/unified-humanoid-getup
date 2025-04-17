import matplotlib.pyplot as plt

# Data for Similar vs. Diverse Morph experiment
conditions = ["Similar", "Diverse"]
success_rates = [0.53, 0.00]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(conditions, success_rates)
ax.set_ylim(0, 1)
ax.set_ylabel("Zero-Shot Success Rate on sig")
ax.set_title("Effect of Morphological Diversity\n(3 training morphs)")
# Annotate bars
for bar, rate in zip(bars, success_rates):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{rate:.2f}", ha='center')
plt.tight_layout()
plt.show()
