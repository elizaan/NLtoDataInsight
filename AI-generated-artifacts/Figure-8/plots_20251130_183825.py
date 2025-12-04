import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import json

# Paths
cache_path = '/Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app/ai_data/data_cache/dyamond_llc2160/data_20251130_183825.npz'
plots_dir = '/Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app/ai_data/plots/dyamond_llc2160'
os.makedirs(plots_dir, exist_ok=True)

# Load cached data
data = np.load(cache_path, allow_pickle=True)
print("Available keys:", data.files)

# Extract arrays and metadata safely
def get_or_none(key):
    return data[key] if key in data else None

temperature = get_or_none('temperature_surface')  # shape: (ntime, ny, nx)
u = get_or_none('u_surface')                      # shape: (ntime, ny, nx)
v = get_or_none('v_surface')                      # shape: (ntime, ny, nx)
mask2d = get_or_none('ocean_mask_surface')        # shape: (ny, nx), 1 for ocean

timesteps = get_or_none('timesteps')
datetimes = get_or_none('datetimes')
x_range = get_or_none('x_range')
y_range = get_or_none('y_range')
z_range = get_or_none('z_range')
lat_range = get_or_none('lat_range')
lon_range = get_or_none('lon_range')
quality_arr = get_or_none('quality_level')
strategy = get_or_none('strategy')
results_json = get_or_none('results_json')

# Basic validations
if temperature is None or u is None or v is None:
    raise RuntimeError("Missing required arrays in cache: temperature_surface, u_surface, v_surface")

ntime, ny, nx = temperature.shape
quality = int(quality_arr[0]) if quality_arr is not None else 0
lat_min, lat_max = (float(lat_range[0]), float(lat_range[1])) if lat_range is not None else (0.0, float(ny))
lon_min, lon_max = (float(lon_range[0]), float(lon_range[1])) if lon_range is not None else (0.0, float(nx))

# Build coordinate arrays for plotting (approximate lon/lat extents; underlying grid is curvilinear)
# For heatmap: use imshow with origin='lower' and explicit extent
extent = [lon_min, lon_max, lat_min, lat_max]

# Determine consistent color scale across months
temp_vmin = float(np.nanmin(temperature))
temp_vmax = float(np.nanmax(temperature))

# Subsample velocity for quiver to reduce clutter
step_y = 8
step_x = 8
y_idx = np.arange(0, ny, step_y)
x_idx = np.arange(0, nx, step_x)

# Coordinates for quiver (approximate lon/lat)
lon_lin = np.linspace(lon_min, lon_max, nx)
lat_lin = np.linspace(lat_min, lat_max, ny)
Lon_sub, Lat_sub = np.meshgrid(lon_lin[x_idx], lat_lin[y_idx])

# Retroreflection focus box (approximate location near Cape Agulhas)
retro_box = {
    "lon_min": 16.5,
    "lon_max": 22.5,
    "lat_min": -40.0,
    "lat_max": -36.0
}

# Optional: parse stats for annotations
stats = None
if results_json is not None and len(results_json) > 0:
    try:
        stats = json.loads(results_json[0])
    except Exception:
        stats = None

# Create one plot per month: temperature heatmap + velocity quiver overlay
num_plots = ntime
for i in range(num_plots):
    temp2d = temperature[i]
    u2d = u[i]
    v2d = v[i]

    # Subsample vectors
    U_sub = u2d[np.ix_(y_idx, x_idx)]
    V_sub = v2d[np.ix_(y_idx, x_idx)]

    # Compute a representative speed percentile for quiver key
    speed2d = np.sqrt(u2d**2 + v2d**2)
    spd95 = float(np.nanpercentile(speed2d, 95.0)) if np.isfinite(speed2d).any() else 1.0
    ref_speed = 1.0  # m/s for key

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    im = ax.imshow(
        temp2d,
        origin='lower',
        extent=extent,
        cmap='plasma',
        vmin=temp_vmin,
        vmax=temp_vmax,
        interpolation='nearest',
        aspect='equal'
    )

    # Overlay velocity vectors (auto scale)
    Q = ax.quiver(
        Lon_sub, Lat_sub, U_sub, V_sub,
        color='k', angles='xy', scale_units='xy', scale=None, width=0.002, alpha=0.85
    )
    try:
        ax.quiverkey(Q, 0.87, 1.03, ref_speed, f'{ref_speed:.1f} m/s',
                     labelpos='E', coordinates='axes')
    except Exception:
        pass

    # Draw retroreflection focus box
    box_w = retro_box["lon_max"] - retro_box["lon_min"]
    box_h = retro_box["lat_max"] - retro_box["lat_min"]
    rect = Rectangle(
        (retro_box["lon_min"], retro_box["lat_min"]),
        box_w, box_h, linewidth=1.2, edgecolor='cyan', facecolor='none', alpha=0.9
    )
    ax.add_patch(rect)
    ax.text(retro_box["lon_min"] + 0.1, retro_box["lat_max"] - 0.3,
            "Agulhas retroflection (approx.)",
            color='cyan', fontsize=9, ha='left', va='top',
            bbox=dict(facecolor='black', alpha=0.25, edgecolor='none', pad=1.5))

    # Labels and title
    date_str = str(datetimes[i]) if datetimes is not None else f"timestep={timesteps[i] if timesteps is not None else i}"
    ax.set_title(f"Agulhas retroflection: Temperature and surface currents\n{date_str} | z=0 | Q={quality}", fontsize=12)
    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg N)")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Temperature (°C)")

    # Optionally annotate stats
    if stats and i < len(stats):
        s = stats[i]
        txt = f"T mean={s['temperature_mean']:.2f}°C, min={s['temperature_min']:.2f}, max={s['temperature_max']:.2f} | " \
              f"|U| mean={s['speed_mean']:.2f} m/s, max={s['speed_max']:.2f}"
        ax.text(0.01, -0.10, txt, transform=ax.transAxes, fontsize=9, color='dimgray')

    # Save
    out_path = os.path.join(plots_dir, f'plot_{i+1}_20251130_183825.png')
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")

print(f"Created {num_plots} plots successfully")