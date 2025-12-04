import numpy as np
import matplotlib.pyplot as plt

# Load cached data
data_path = '/Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app/ai_data/data_cache/dyamond_llc2160/data_20251127_124827.npz'
data = np.load(data_path)

# Inspect available keys
print("Available keys:", data.files)

# Extract data
velocity_magnitude = data['velocity_magnitude']

# Plot 1: 2D Spatial Field Visualization of Velocity Magnitude with Geographic Context
plt.figure(figsize=(12, 8))
plt.imshow(velocity_magnitude[:, :, 0], origin='lower', cmap='viridis', extent=[0, 8640, 0, 6480])
plt.colorbar(label='Velocity Magnitude (m/s)')
plt.title('Velocity Magnitude on January 20, 2020 (Surface Level, Moderate Quality)\nPlotted at reduced resolution (quality=-q) but coordinates show original grid')
plt.xlabel('X Index (approx. longitude)')
plt.ylabel('Y Index (approx. latitude)')

# Adding latitude and longitude labels
plt.xticks(ticks=np.linspace(0, 8640, num=5), labels=np.round(np.linspace(-180, 180, num=5), 2))  # Example longitude labels
plt.yticks(ticks=np.linspace(0, 6480, num=5), labels=np.round(np.linspace(-90, 90, num=5), 2))  # Example latitude labels

plt.savefig('/Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app/ai_data/plots/dyamond_llc2160/plot_1_20251127_124827_revised.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: 1D Time Series of Velocity Magnitude with Improved Binning
plt.figure(figsize=(10, 6))
plt.hist(velocity_magnitude.flatten(), bins=100, color='blue', alpha=0.7)  # Increased number of bins for better insight
plt.title('Histogram of Velocity Magnitude on January 20, 2020 (Moderate Quality)')
plt.xlabel('Velocity Magnitude (m/s)')
plt.ylabel('Frequency')
plt.xlim(0, np.max(velocity_magnitude))  # Set x-axis limit to max velocity magnitude
plt.ylim(0, np.max(np.histogram(velocity_magnitude.flatten(), bins=100)[0]) * 1.1)  # Set y-axis limit slightly above max frequency

plt.savefig('/Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app/ai_data/plots/dyamond_llc2160/plot_2_20251127_124827_revised.png', dpi=150, bbox_inches='tight')
plt.close()

print("Created 2 revised plots successfully")