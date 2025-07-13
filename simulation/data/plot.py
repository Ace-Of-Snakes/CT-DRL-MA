import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the KDE model
with open('models/kde_model_dwelltime_hours.pkl', 'rb') as f:
    kde_model = pickle.load(f)

# Extract the original dataset to determine the appropriate range
original_data = kde_model.dataset.flatten()

# Get data statistics
data_min = original_data.min()
data_max = original_data.max()
data_mean = original_data.mean()
data_std = original_data.std()

print(f"Data range: {data_min:.2f} to {data_max:.2f} hours")
print(f"Mean: {data_mean:.2f} hours, Std: {data_std:.2f} hours")

# Create x-axis range for plotting
# Add some padding beyond the data range for better visualization
padding = (data_max - data_min) * 0.1
x_min = max(0, data_min - padding)  # Don't go below 0 for dwell time
x_max = data_max + padding

# Create dense x values for smooth curve
x_plot = np.linspace(x_min, x_max, 1000)

# Evaluate the KDE at these points
pdf_values = kde_model.pdf(x_plot)

# Create the plot
plt.figure(figsize=(12, 8))

# Plot the KDE
plt.plot(x_plot, pdf_values, 'b-', linewidth=2, label='KDE (Probability Density)')

# Optionally, overlay histogram of original data for comparison
plt.hist(original_data, bins=50, density=True, alpha=0.3, color='gray', 
         label='Original Data Histogram')

# Formatting
plt.xlabel('Dwell Time (hours)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.title('KDE Model: Dwell Time Distribution', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Add some statistics as text
stats_text = f'n = {kde_model.n:,}\nMean = {data_mean:.1f}h\nStd = {data_std:.1f}h'
plt.text(0.75, 0.75, stats_text, transform=plt.gca().transAxes, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
         verticalalignment='top', fontsize=10)

plt.tight_layout()
plt.show()

# Optional: Create a second plot focusing on the main distribution
# (excluding extreme outliers for better visualization)
plt.figure(figsize=(12, 6))

# Focus on central 95% of data
q025 = np.percentile(original_data, 2.5)
q975 = np.percentile(original_data, 97.5)
x_focused = np.linspace(q025, q975, 1000)
pdf_focused = kde_model.pdf(x_focused)

plt.plot(x_focused, pdf_focused, 'r-', linewidth=2)
plt.fill_between(x_focused, pdf_focused, alpha=0.3, color='red')

plt.xlabel('Dwell Time (hours)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.title('KDE Model: Dwell Time Distribution (Central 95%)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Add percentile markers
plt.axvline(data_mean, color='black', linestyle='--', alpha=0.7, label=f'Mean ({data_mean:.1f}h)')
plt.axvline(np.median(original_data), color='green', linestyle='--', alpha=0.7, 
           label=f'Median ({np.median(original_data):.1f}h)')
plt.legend()

plt.tight_layout()
plt.show()

# Print some useful statistics
print(f"\nBandwidth factor: {kde_model.factor:.6f}")
print(f"Covariance: {kde_model.covariance[0,0]:.2f}")
print(f"Data percentiles:")
print(f"  25th: {np.percentile(original_data, 25):.1f}h")
print(f"  50th: {np.percentile(original_data, 50):.1f}h") 
print(f"  75th: {np.percentile(original_data, 75):.1f}h")
print(f"  95th: {np.percentile(original_data, 95):.1f}h")