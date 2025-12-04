import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# Improved Plot 1: Heatmap of Temperature changes over 14 months
plt.figure(figsize=(12, 8))
for i in range(temperature_data.shape[0]):
    ax = plt.subplot(4, 4, i + 1)
    sns.heatmap(temperature_data[i, 0], cmap='coolwarm', cbar=True,
    xticklabels=False, yticklabels=False,
    vmin=np.min(temperature_data), vmax=np.max(temperature_data),ax=ax)
    ax.invert_yaxis()
    ax.set_title(f'Month {i + 1}')
plt.suptitle('Temperature Distribution in Agulhas Region Over 14 Months', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('/Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app/ai_data/plots/dyamond_llc2160/plot_1_20251121_140122.png', dpi=150, bbox_inches='tight')
plt.close()

# Improved Plot 2: Velocity field visualization with annotations for retroflection
for i in range(len(eastwest_velocity_data)):
    plt.figure(figsize=(12, 6))
    eastwest_vel = eastwest_velocity_data[i, 0]
    northsouth_vel = northsouth_velocity_data[i, 0]
    y_dim, x_dim = eastwest_vel.shape
    X, Y = np.meshgrid(np.arange(x_dim), np.arange(y_dim))
    
    plt.quiver(X, Y, eastwest_vel, northsouth_vel, scale=50, headwidth=3)
    plt.title(f'Current Direction in Agulhas Region - Month {i + 1}')
    plt.xlabel('X Index')
    plt.ylabel('Y Index')

    # Annotate areas of retroflection (example coordinates, adjust as needed)
    retroflection_coords = [(10, 20), (15, 25), (20, 30)]  # Example coordinates
    for (x, y) in retroflection_coords:
        plt.annotate('Retroflection', xy=(x, y), xytext=(x + 2, y + 2),
                     arrowprops=dict(facecolor='red', shrink=0.05))

    plt.savefig(f'/Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app/ai_data/plots/dyamond_llc2160/plot_2_{i + 1}_20251121_140122.png', dpi=150, bbox_inches='tight')
    plt.close()

print("Created improved plots successfully")