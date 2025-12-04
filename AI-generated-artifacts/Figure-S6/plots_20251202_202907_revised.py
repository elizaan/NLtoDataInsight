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

# Build a depth (m) axis for z-levels using available info; fallback to linear span of z_range
# Try a few common keys; otherwise linearly map z_range to nz levels
depth_axis_m = None
for k in ['depth_levels_m', 'depth_m', 'z_levels_m', 'z_depths_m', 'z_depths', 'depths']:
    if k in data.files:
        depth_axis_m = np.array(data[k]).astype(float).ravel()
        break
if depth_axis_m is None or depth_axis_m.size != nz:
    # Fallback: assume z_range provides min/max depth in meters; convert to positive-down and ascending
    zr0, zr1 = float(z_range[0]), float(z_range[1])
    # Use absolute values and ensure ascending positive depth
    d_min = min(abs(zr0), abs(zr1))
    d_max = max(abs(zr0), abs(zr1))
    # If range includes zero, start at 0 for clarity
    if (zr0 <= 0 <= zr1) or (zr1 <= 0 <= zr0) or d_min > 0:
        d_min = 0.0
    depth_axis_m = np.linspace(d_min, d_max, nz)

# Ensure monotonic ascending depth (0 -> max)
if depth_axis_m[0] > depth_axis_m[-1]:
    depth_axis_m = depth_axis_m[::-1]
depth_min_m = float(np.nanmin(depth_axis_m))
depth_max_m = float(np.nanmax(depth_axis_m))

# Color scales: use a percentile-based "saturated" scale plus a full-range scale
valid_vals = velmag[np.isfinite(velmag)]
if valid_vals.size > 0:
    p95 = float(np.nanpercentile(valid_vals, 95.0))
    p99 = float(np.nanpercentile(valid_vals, 99.0))
else:
    p95 = overall_max
    p99 = overall_max

vmin = 0.0
vmax_sat = max(p99, 1e-6)  # percentile-based vmax (99th)
vmax_full = max(overall_max, vmax_sat)  # full-range vmax (no saturation)
cmap = cm.viridis
norm_sat = colors.Normalize(vmin=vmin, vmax=vmax_sat)
norm_full = colors.Normalize(vmin=vmin, vmax=vmax_full)
mappable_sat = cm.ScalarMappable(norm=norm_sat, cmap=cmap)
mappable_sat.set_array([])
mappable_full = cm.ScalarMappable(norm=norm_full, cmap=cmap)
mappable_full.set_array([])

# Output plots directory
plot_dir = '/Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app/ai_data/plots/dyamond_llc2160'
os.makedirs(plot_dir, exist_ok=True)

# -------------------------------------------------------
# Plot 1: "3D" stacked textured surfaces with true depth (m) and vertical exaggeration
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

# Vertical exaggeration (relative to meters for plotting aesthetics)
vert_exag = 20.0  # stated exaggeration
# Render shallower levels on top by plotting from deep to shallow
for k in sorted(slice_indices, reverse=True):
    S = velmag[k, ::stride_y, ::stride_x]  # (ny_ds, nx_ds)
    # Place Z using true depth (m), plotted downward (negative), scaled by vertical exaggeration
    Z = - (depth_axis_m[k] * vert_exag) * np.ones_like(S)
    facecolors = cmap(norm_sat(S))
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=facecolors, linewidth=0, antialiased=False, shade=False, alpha=0.95)

# Axis labels and limits
ax.set_xlabel('Longitude (°)')
ax.set_ylabel('Latitude (°)')
ax.set_zlabel('Depth (m)')
ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)

# Depth axis ticks: show as positive meters while data plotted as negative
tick_depths = np.linspace(0.0, depth_max_m, num=6)
ax.set_zticks(-tick_depths * vert_exag)
ax.set_zticklabels([f'{int(td)}' for td in tick_depths])
ax.set_zlim(-depth_max_m * vert_exag, 0.0)

# Add colorbar with annotation of vmax
cbar = fig.colorbar(mappable_sat, ax=ax, pad=0.02, shrink=0.7)
cbar.set_label(f'Speed |u| (m/s)\ncolors saturate at p99={vmax_sat:.2f}; max={overall_max:.2f}')

# Titles/caption
title = (
    f'Gulf of Mexico velocity magnitude (stacked textured surfaces)\n'
    f'Timestep: {timestep_datetime} UTC | Q={quality} | lat [{lat_min:.2f}, {lat_max:.2f}] | lon [{lon_min:.2f}, {lon_max:.2f}]'
)
ax.set_title(title, pad=20)

caption = (
    f'Using true depths (m) with {vert_exag:.0f}x vertical exaggeration; '
    f'slices at z-indices: {slice_indices}. Downsampled grid: stride_x={stride_x}, stride_y={stride_y}.'
)
fig.text(0.5, 0.02, caption, ha='center', va='bottom', fontsize=10)

plot1_path = os.path.join(plot_dir, 'plot_1_20251202_202907_revised.png')
plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
plt.close(fig)

# -------------------------------------------------------
# Plot 2: Vertical cross-sections along two transects (Depth in meters, 0 m at top)
# -------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

# Helper: imshow with origin='lower' and explicit extent in meters
def plot_section(ax, section, x_coords, xlabel, title, depth_m, norm_used):
    # section shape: (nz, nX); depth_m ascending positive (0 -> max)
    extent = [float(x_coords[0]), float(x_coords[-1]), float(depth_m[0]), float(depth_m[-1])]
    im = ax.imshow(section, aspect='auto', origin='lower', extent=extent, cmap=cmap, norm=norm_used)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Depth (m)')
    ax.set_title(title)
    # Invert so 0 m appears at the top
    ax.set_ylim(depth_m[-1], depth_m[0])
    return im

# West–East across 24.5°N
lat_transect = 24.5
y_idx = int(np.clip(round((lat_transect - lat_min) / (lat_max - lat_min) * (ny - 1)), 0, ny - 1))
sec_WE = velmag[:, y_idx, :]  # (z, x)
im1 = plot_section(
    axes[0], sec_WE, lons, xlabel='Longitude (°)',
    title=f'W–E section at ~{lat_transect}°N (y_idx={y_idx})',
    depth_m=depth_axis_m, norm_used=norm_sat
)

# North–South along 90°W
lon_transect = -90.0
x_idx = int(np.clip(round((lon_transect - lon_min) / (lon_max - lon_min) * (nx - 1)), 0, nx - 1))
sec_NS = velmag[:, :, x_idx]  # (z, y)
im2 = plot_section(
    axes[1], sec_NS, lats, xlabel='Latitude (°)',
    title=f'N–S section at ~{lon_transect}° (x_idx={x_idx})',
    depth_m=depth_axis_m, norm_used=norm_sat
)

# Shared colorbar with annotation
cbar = fig.colorbar(mappable_sat, ax=axes.ravel().tolist(), shrink=0.9, pad=0.02)
cbar.set_label(f'Speed |u| (m/s)\ncolors saturate at p99={vmax_sat:.2f}; max={overall_max:.2f}')

fig.suptitle(
    f'Velocity magnitude vertical cross-sections | Timestep: {timestep_datetime} UTC | Q={quality}\n'
    f'lat [{lat_min:.2f}, {lat_max:.2f}], lon [{lon_min:.2f}, {lon_max:.2f}] | Depth range: 0–{depth_max_m:.0f} m',
    fontsize=12
)
plot2_path = os.path.join(plot_dir, 'plot_2_20251202_202907_revised.png')
plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
plt.close(fig)

# -------------------------------------------------------
# Plot 3: Distributions – histograms (linear + log) and depthwise percentiles with per-depth stats
# -------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Histogram (overall) – linear scale spanning full [0, 2.48] m/s
valid_vals = velmag[np.isfinite(velmag)]
bin_max = 2.48
bins = np.linspace(0.0, bin_max, 100)
axes[0].hist(valid_vals, bins=bins, color='steelblue', edgecolor='none')
axes[0].set_xlabel('Speed |u| (m/s)')
axes[0].set_ylabel('Voxel count')
axes[0].set_title('Velocity magnitude histogram (linear scale)')
axes[0].set_xlim(0.0, bin_max)

# Histogram (overall) – log-scaled y-axis for coverage
axes[1].hist(valid_vals, bins=bins, color='steelblue', edgecolor='none')
axes[1].set_yscale('log')
axes[1].set_xlabel('Speed |u| (m/s)')
axes[1].set_ylabel('Voxel count (log)')
axes[1].set_title('Velocity magnitude histogram (log scale)')
axes[1].set_xlim(0.0, bin_max)

# Draw percentile lines on both histograms
percentiles_to_mark = [5, 25, 50, 75, 95, 99]
pct_values = {}
for p in percentiles_to_mark:
    if overall_percentiles is not None and len(overall_percentiles) >= 5 and p in [5, 25, 50, 75, 95]:
        # Map index in provided percentiles
        mapping = {5:0, 25:1, 50:2, 75:3, 95:4}
        val = float(overall_percentiles[mapping[p]])
    else:
        val = float(np.nanpercentile(valid_vals, p)) if valid_vals.size > 0 else np.nan
    pct_values[p] = val

colors_p = {5: '#a6cee3', 25: '#1f78b4', 50: '#33a02c', 75: '#fb9a1c', 95: '#e31a1c', 99: '#6a3d9a'}
for axh in [axes[0], axes[1]]:
    for p, val in pct_values.items():
        if np.isfinite(val):
            axh.axvline(val, color=colors_p.get(p, 'k'), linestyle='--', linewidth=1, alpha=0.9, label=f'p{p}={val:.2f}')
    # Build a compact legend without duplicates
    handles, labels = axh.get_legend_handles_labels()
    # Deduplicate by label
    uniq = {}
    for h, lb in zip(handles, labels):
        uniq[lb] = h
    axh.legend(uniq.values(), uniq.keys(), title='Percentiles', fontsize=8, loc='upper right')

# Depthwise percentiles and per-depth stats
ax_stats = axes[2]
if per_depth_p is not None:
    # Percentiles
    for i, p in enumerate(pcts.tolist()):
        ax_stats.plot(per_depth_p[:, i], depth_axis_m, label=f'{p}th', color=colors_p.get(int(p), None), linewidth=2)
else:
    ax_stats.text(0.5, 0.5, 'Percentile-by-depth data not available', ha='center', va='center')

# Overlay per-depth stats if available
if per_depth_mean is not None:
    ax_stats.plot(per_depth_mean, depth_axis_m, color='k', linewidth=2, label='mean')
if per_depth_std is not None and per_depth_mean is not None:
    mean = np.asarray(per_depth_mean).ravel()
    std = np.asarray(per_depth_std).ravel()
    low = mean - std
    high = mean + std
    ax_stats.fill_betweenx(depth_axis_m, low, high, color='k', alpha=0.15, label='mean ± 1σ')
if per_depth_min is not None:
    ax_stats.plot(per_depth_min, depth_axis_m, color='gray', linestyle='--', linewidth=1.5, label='min')
if per_depth_max is not None:
    ax_stats.plot(per_depth_max, depth_axis_m, color='gray', linestyle='--', linewidth=1.5, label='max')

ax_stats.invert_yaxis()  # 0 m at top
ax_stats.set_xlabel('Speed |u| (m/s)')
ax_stats.set_ylabel('Depth (m)')
ax_stats.grid(True, alpha=0.3)
ax_stats.legend(title='Depthwise stats', fontsize=8)
ax_stats.set_title('Depthwise percentiles and stats')

fig.suptitle(
    f'Distributions of velocity magnitude | Timestep: {timestep_datetime} UTC | Q={quality}\n'
    f'Overall: min={overall_min:.3f}, max={overall_max:.3f}, mean={overall_mean:.3f}, std={overall_std:.3f}',
    fontsize=12
)
plot3_path = os.path.join(plot_dir, 'plot_3_20251202_202907_revised.png')
plt.savefig(plot3_path, dpi=150, bbox_inches='tight')
plt.close(fig)

# -------------------------------------------------------
# Plot 4 (new): Cross-sections without saturation (full-range color scale)
# -------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

# West–East across 24.5°N (full range)
im1_full = axes[0].imshow(
    sec_WE, aspect='auto', origin='lower',
    extent=[float(lons[0]), float(lons[-1]), float(depth_axis_m[0]), float(depth_axis_m[-1])],
    cmap=cmap, norm=norm_full
)
axes[0].set_xlabel('Longitude (°)')
axes[0].set_ylabel('Depth (m)')
axes[0].set_title(f'W–E section (no saturation) at ~{lat_transect}°N (y_idx={y_idx})')
axes[0].set_ylim(depth_axis_m[-1], depth_axis_m[0])

# North–South along 90°W (full range)
im2_full = axes[1].imshow(
    sec_NS, aspect='auto', origin='lower',
    extent=[float(lats[0]), float(lats[-1]), float(depth_axis_m[0]), float(depth_axis_m[-1])],
    cmap=cmap, norm=norm_full
)
axes[1].set_xlabel('Latitude (°)')
axes[1].set_ylabel('Depth (m)')
axes[1].set_title(f'N–S section (no saturation) at ~{lon_transect}° (x_idx={x_idx})')
axes[1].set_ylim(depth_axis_m[-1], depth_axis_m[0])

cbar_full = fig.colorbar(mappable_full, ax=axes.ravel().tolist(), shrink=0.9, pad=0.02)
cbar_full.set_label(f'Speed |u| (m/s)\nfull range up to max={overall_max:.2f}')

fig.suptitle(
    f'Velocity magnitude cross-sections (full color range) | Timestep: {timestep_datetime} UTC | Q={quality}',
    fontsize=12
)
plot4_path = os.path.join(plot_dir, 'plot_4_20251202_202907_revised.png')
plt.savefig(plot4_path, dpi=150, bbox_inches='tight')
plt.close(fig)

# -------------------------------------------------------
# Report summary to stdout
# -------------------------------------------------------
summary = {
    "plots_created": 4,
    "plot_paths": [plot1_path, plot2_path, plot3_path, plot4_path],
    "timestep_datetime": timestep_datetime,
    "timestep_index": timestep_index,
    "domain_lat_range": lat_range,
    "domain_lon_range": lon_range,
    "domain_depth_range_m": [depth_min_m, depth_max_m],
    "quality_level": quality,
    "overall_stats": {
        "min": overall_min, "max": overall_max, "mean": overall_mean, "std": overall_std
    },
    "notes": (
        "All vertical axes use physical depth (m). Cross-sections invert y so 0 m is at the top. "
        f"3D surfaces placed at true depths with {vert_exag:.0f}x vertical exaggeration. "
        f"Default color scale saturates at p99={vmax_sat:.2f} m/s; true max={overall_max:.2f} m/s. "
        "Plot 4 provides full-range color mapping without saturation. Histograms span [0, 2.48] m/s with linear and log scales; percentile lines included. "
        "Depthwise panel overlays min/max/mean and mean±1σ on percentile curves."
    )
}
print(json.dumps(summary, indent=2))