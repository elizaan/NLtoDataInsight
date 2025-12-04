import numpy as np
import matplotlib.pyplot as plt

# Load cached data
data = np.load('/Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app/ai_data/data_cache/dyamond_llc2160/data_20251121_140122.npz')

# Inspect available keys
print("Available keys:", data.files)

# Extract data
temperature_data = data['temperature']
eastwest_velocity_data = data['eastwest_velocity']
northsouth_velocity_data = data['northsouth_velocity']
x_range = data['x_range']
y_range = data['y_range']
quality = data['quality']

# Check the shape of the data arrays
print("Temperature data shape:", temperature_data.shape)
print("East-west velocity data shape:", eastwest_velocity_data.shape)
print("North-south velocity data shape:", northsouth_velocity_data.shape)

# Plot 1: Temperature changes over 14 months
plt.figure(figsize=(12, 6))
for i, temp in enumerate(temperature_data):
    plt.plot(temp[0].flatten(), label=f'Month {i+1}')  # Use the first depth slice
plt.title('Temperature Changes in Agulhas Region Over 14 Months')
plt.xlabel('Grid Points (Flattened)')
plt.ylabel('Temperature (°C)')
plt.legend(loc='upper right')
plt.savefig('/Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app/ai_data/plots/dyamond_llc2160/plot_1_20251121_140122.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: Velocity field visualization
for i in range(len(eastwest_velocity_data)):
    plt.figure(figsize=(12, 6))
    # Use the first depth slice
    eastwest_vel = eastwest_velocity_data[i, 0]
    northsouth_vel = northsouth_velocity_data[i, 0]
    # Create a grid for quiver plot
    y_dim, x_dim = eastwest_vel.shape
    X, Y = np.meshgrid(np.arange(x_dim), np.arange(y_dim))
    plt.quiver(X, Y, eastwest_vel, northsouth_vel, scale=50, headwidth=3)
    plt.title(f'Current Direction in Agulhas Region - Month {i+1}')
    plt.xlabel('X Index')
    plt.ylabel('Y Index')
    plt.savefig(f'/Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app/ai_data/plots/dyamond_llc2160/plot_2_{i+1}_20251121_140122.png', dpi=150, bbox_inches='tight')
    plt.close()

print("Created 15 plots successfully")