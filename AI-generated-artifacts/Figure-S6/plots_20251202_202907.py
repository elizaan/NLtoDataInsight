import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors

# -------------------------
# Load cached data
# -------------------------
npz_path = '/Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app/ai_data/data_cache/dyamond_llc2160/data_20251202_202907.npz'
data = np.load(npz_path)

print("Available keys in NPZ:", data.files)

# Extract arrays/metadata
velmag = data['velocity_magnitude']  # shape (Z, Y, X)
x_range = data['x_range'].tolist()
y_range = data['y_range'].tolist()
z_range = data['z_range'].tolist()
lat_range = data['lat_range'].tolist()
lon_range = data['lon_range'].tolist()
quality = int(data['quality_level'][0]) if 'quality_level' in data.files else 0
timestep_index = int(data['timestep_index'][0])
timestep_datetime = str(data['timestep_datetime'][0]) if 'timestep_datetime' in data.files else 'unknown'

overall_min = float(data['overall_min'][0])
overall_max = float(data['overall_max'][0])
overall_mean = float(data['overall_mean'][0])
overall_std = float(data['overall_std'][0])
overall_percentiles = data['overall_percentiles'] if 'overall_percentiles' in data.files else None
per_depth_min = data['per_depth_min'] if 'per_depth_min' in data.files else None
per_depth_max = data['per_depth_max'] if 'per_depth_max' in data.files else None
per_depth_mean = data['per_depth_mean'] if 'per_depth_mean' in data.files else None
per_depth_std = data['per_depth_std'] if 'per_depth_std' in data.files else None
per_depth_p = data['per_depth_percentiles'] if 'per_depth_percentiles' in data.files else None
pcts = data['percentiles_list'] if 'percentiles_list' in data.files else np.array([5,25,50,75,95], dtype=np.int32)

hist_counts = data['histogram_counts'] if 'histogram_counts' in data.files else None
hist_edges = data['histogram_bin_edges'] if 'histogram_bin_edges' in data.files else None

# Shapes and coordinate helpers
nz, ny, nx = velmag.shape
lat_min, lat_max = lat_range
lon_min, lon_max = lon_range

# Construct approximate lon/lat coordinates (linear across subdomain; LLC grid is curvilinear, so this is an approximation)
lons = np.linspace(lon_min, lon_max, nx)
lats = np.linspace(lat_min, lat_max, ny)
z_levels = np.arange(nz)  # 0=surface index

# Color scale: use 95th percentile as vmax to emphasize structure, avoid outliers
if overall_percentiles is not None and overall_percentiles.size >= 5:
    vmin = 0.0
    vmax = float(overall_percentiles[4])  # 95th percentile
else:
    # fallback: robust upper bound at 99.5th percentile computed on the fly
    valid_vals = velmag[np.isfinite(velmag)]
    vmax = float(np.nanpercentile(valid_vals, 95.0)) if valid_vals.size > 0 else overall_max
    vmin = 0.0

cmap = cm.viridis
norm = colors.Normalize(vmin=vmin, vmax=vmax)
mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
mappable.set_array([])

# Output plots directory
plot_dir = '/Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app/ai_data/plots/dyamond_llc2160'
os.makedirs(plot_dir, exist_ok=True)

# -------------------------------------------------------
# Plot 1: "3D" stacked textured surfaces with exaggeration
# -------------------------------------------------------
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Downsample for plotting speed
stride_x = max(1, nx // 150)
stride_y = max(1, ny // 150)

X, Y = np.meshgrid(lons[::stride_x], lats[::stride_y])  # (ny_ds, nx_ds)

# Choose a handful of depth slices
slice_indices = [0, 5, 10, 15, 20, 30, 45, 60]
slice_indices = [k for k in slice_indices if k < nz]

# Vertical exaggeration (unitless because we use z-index)
vert_exag = 20.0
dz = 1.0  # per z-level index
# Render shallower levels on top by plotting from deep to shallow
for k in sorted(slice_indices, reverse=True):
    S = velmag[k, ::stride_y, ::stride_x]  # (ny_ds, nx_ds)
    # Z coordinate: negative going down, exaggerated
    Z = - (k * dz * vert_exag) * np.ones_like(S)
    # Face colors based on magnitude
    facecolors = cmap(norm(S))
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=facecolors, linewidth=0, antialiased=False, shade=False, alpha=0.95)

# Axis labels and limits
ax.set_xlabel('Longitude (°)')
ax.set_ylabel('Latitude (°)')
ax.set_zlabel('Relative depth (z-index, exaggerated)')
ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)
# Set z-limits to cover plotted slices
zmin = -max(slice_indices) * dz * vert_exag if slice_indices else -nz * dz * vert_exag
zmax = 0.0
ax.set_zlim(zmin, zmax)

# Add colorbar
cbar = fig.colorbar(mappable, ax=ax, pad=0.02, shrink=0.7)
cbar.set_label('Speed |u| (m/s)\n(colors saturate at ~95th pct)')

# Titles/caption
title = f'Gulf of Mexico velocity magnitude (stacked textured surfaces)\nTimestep: {timestep_datetime} UTC | Q={quality} | lat [{lat_min:.2f}, {lat_max:.2f}] | lon [{lon_min:.2f}, {lon_max:.2f}]'
ax.set_title(title, pad=20)

caption = (
    f'Vertical exaggeration: {vert_exag:.0f}x (relative z-index). '
    f'Notable: fast surface currents (up to ~{vmax:.2f} m/s in 95th pct scale) with rapid decay with depth; '
    f'quiescent interior below mid-depths.'
)
fig.text(0.5, 0.02, caption, ha='center', va='bottom', fontsize=10)

plot1_path = os.path.join(plot_dir, 'plot_1_20251202_202907.png')
plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
plt.close(fig)

# -------------------------------------------------------
# Plot 2: Vertical cross-sections along two transects
# -------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

# Helper: imshow with origin='lower' and explicit extent; row-0 is bottom by convention here
def plot_section(ax, section, x_coords, xlabel, title):
    # section shape: (nz, nX)
    extent = [float(x_coords[0]), float(x_coords[-1]), 0.0, float(nz - 1)]
    im = ax.imshow(section, aspect='auto', origin='lower', extent=extent, cmap=cmap, norm=norm)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('z-level (0=surface at bottom)')
    ax.set_title(title)
    return im

# West–East across 24.5°N
lat_transect = 24.5
y_idx = int(np.clip(round((lat_transect - lat_min) / (lat_max - lat_min) * (ny - 1)), 0, ny - 1))
sec_WE = velmag[:, y_idx, :]  # (z, x)
im1 = plot_section(
    axes[0], sec_WE, lons, xlabel='Longitude (°)',
    title=f'W–E section at ~{lat_transect}°N (y_idx={y_idx})'
)

# North–South along 90°W
lon_transect = -90.0
x_idx = int(np.clip(round((lon_transect - lon_min) / (lon_max - lon_min) * (nx - 1)), 0, nx - 1))
sec_NS = velmag[:, :, x_idx]  # (z, y)
im2 = plot_section(
    axes[1], sec_NS, lats, xlabel='Latitude (°)',
    title=f'N–S section at ~{lon_transect}° (x_idx={x_idx})'
)

# Shared colorbar
cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), shrink=0.9, pad=0.02)
cbar.set_label('Speed |u| (m/s)')

fig.suptitle(
    f'Velocity magnitude vertical cross-sections | Timestep: {timestep_datetime} UTC | Q={quality}\n'
    f'lat [{lat_min:.2f}, {lat_max:.2f}], lon [{lon_min:.2f}, {lon_max:.2f}]',
    fontsize=12
)
plot2_path = os.path.join(plot_dir, 'plot_2_20251202_202907.png')
plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
plt.close(fig)

# -------------------------------------------------------
# Plot 3: Distributions – histogram and depthwise percentile curves
# -------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Histogram (overall)
if hist_counts is not None and hist_edges is not None:
    centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])
    axes[0].bar(centers, hist_counts, width=(hist_edges[1:] - hist_edges[:-1]), align='center', color='steelblue', edgecolor='none')
    axes[0].set_xlabel('Speed |u| (m/s)')
    axes[0].set_ylabel('Voxel count')
    axes[0].set_title('Velocity magnitude histogram (overall)')
    axes[0].set_xlim(hist_edges[0], hist_edges[-1])
else:
    # Fallback: compute a quick histogram
    valid_vals = velmag[np.isfinite(velmag)]
    axes[0].hist(valid_vals, bins=60, color='steelblue', edgecolor='none')
    axes[0].set_xlabel('Speed |u| (m/s)')
    axes[0].set_ylabel('Voxel count')
    axes[0].set_title('Velocity magnitude histogram (overall)')

# Depthwise percentiles
if per_depth_p is not None:
    # pcts is array like [5,25,50,75,95]
    colors_p = {5: '#a6cee3', 25: '#1f78b4', 50: '#33a02c', 75: '#fb9a99', 95: '#e31a1c'}
    for i, p in enumerate(pcts.tolist()):
        axes[1].plot(per_depth_p[:, i], z_levels, label=f'{p}th', color=colors_p.get(int(p), None), linewidth=2)
    axes[1].invert_yaxis()  # make surface (z=0) at top visually
    axes[1].set_xlabel('Speed |u| (m/s)')
    axes[1].set_ylabel('z-level (0=surface)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(title='Percentiles')
    axes[1].set_title('Depthwise percentiles of |u|')
else:
    axes[1].axis('off')
    axes[1].text(0.5, 0.5, 'Percentile-by-depth data not available', ha='center', va='center')

fig.suptitle(
    f'Distributions of velocity magnitude | Timestep: {timestep_datetime} UTC | Q={quality}\n'
    f'Overall: min={overall_min:.3f}, max={overall_max:.3f}, mean={overall_mean:.3f}, std={overall_std:.3f}',
    fontsize=12
)
plot3_path = os.path.join(plot_dir, 'plot_3_20251202_202907.png')
plt.savefig(plot3_path, dpi=150, bbox_inches='tight')
plt.close(fig)

# -------------------------------------------------------
# Report summary to stdout
# -------------------------------------------------------
summary = {
    "plots_created": 3,
    "plot_paths": [plot1_path, plot2_path, plot3_path],
    "timestep_datetime": timestep_datetime,
    "timestep_index": timestep_index,
    "domain_lat_range": lat_range,
    "domain_lon_range": lon_range,
    "quality_level": quality,
    "overall_stats": {
        "min": overall_min, "max": overall_max, "mean": overall_mean, "std": overall_std
    },
    "notes": "Plot 1 uses stacked textured surfaces with vertical exaggeration (20x) to approximate a 3D rendering. Cross-sections follow the convention row-0 at bottom for imshow; the depthwise line plot is inverted so surface appears at top."
}
print(json.dumps(summary, indent=2))