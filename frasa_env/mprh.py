import matplotlib.pyplot as plt

# Base Morph Count Scaling data
training_counts = [1, 3, 5, 6]
success_rates = [0.03, 0.40, 0.20, 0.71]

# Similar vs Diverse data at 3
diverse_success = 0.00
similar_success = 0.53

fig, ax = plt.subplots(figsize=(6, 4))

# Plot base scaling curve
ax.plot(training_counts, success_rates, marker='o', label='Original scaling')

# Annotate the dip at 5 morphs
ax.annotate('Dip at 5 (20%)', xy=(5, 0.20), xytext=(5, 0.35),
            arrowprops=dict(arrowstyle='->'))

# Add diverse-only marker at 3 morphs
ax.scatter([3], [diverse_success], color='red', marker='x', s=100, label='3 Diverse Morphs')
ax.annotate('Diverse set (0%)', xy=(3, 0.00), xytext=(3, 0.15),
            arrowprops=dict(arrowstyle='->', color='red'), color='red')

# Formatting
ax.set_xticks(training_counts)
ax.set_xlabel('Number of Training Morphs')
ax.set_ylabel('Zero-Shot Success Rate on sig')
ax.set_title('Morph Count Scaling with Coverage Annotation')
ax.set_ylim(0, 1)
ax.grid(True)
ax.legend(loc='lower right')

plt.tight_layout()
plt.show()
