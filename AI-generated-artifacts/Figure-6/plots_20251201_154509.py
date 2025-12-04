import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy import ndimage
import os
import json

# Paths
cache_path = '/Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app/ai_data/data_cache/dyamond_llc2160/data_20251201_154509.npz'
plots_dir = '/Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app/ai_data/plots/dyamond_llc2160'
os.makedirs(plots_dir, exist_ok=True)

# Load cached data
data = np.load(cache_path, allow_pickle=True)
print("Available keys:", data.files)

# Extract metadata
x_range = data['x_range'].tolist() if 'x_range' in data else None
y_range = data['y_range'].tolist() if 'y_range' in data else None
z_range = data['z_range'].tolist() if 'z_range' in data else None
lat_range = data['lat_range'].tolist() if 'lat_range' in data else None
lon_range = data['lon_range'].tolist() if 'lon_range' in data else None
quality = int(data['quality_level']) if 'quality_level' in data else -6
timesteps = data['timesteps']
dates = [str(d) for d in data['dates']]
time_range = data['time_range'].tolist() if 'time_range' in data else None

# Extract arrays
u_all = data['u']            # shape (nt, ny, nx)
v_all = data['v']            # shape (nt, ny, nx)
zeta_all = data['zeta']      # shape (nt, ny, nx)
ow_all = data['okubo_weiss'] # shape (nt, ny, nx)
ocean_mask = data['ocean_mask']

nt, ny, nx = zeta_all.shape
lat_min, lat_max = lat_range
lon_min, lon_max = lon_range

# Build lon/lat arrays for plotting and quiver sampling
lons = np.linspace(lon_min, lon_max, nx)
lats = np.linspace(lat_min, lat_max, ny)

# Robust color limits (symmetric) for zeta and OW across all times
def robust_sym_lims(arr, low=2.0, high=98.0):
    p = np.nanpercentile(np.abs(arr), high)
    return (-p, p)

zeta_vmin, zeta_vmax = robust_sym_lims(zeta_all, 2.0, 98.0)
ow_vmin, ow_vmax = robust_sym_lims(ow_all, 2.0, 98.0)

# Common extent and origin rules
extent = [lon_min, lon_max, lat_min, lat_max]
origin = 'lower'

# Plot 1: Surface relative vorticity with velocity quivers overlaid (first daily frame)
t_idx = 0  # 2020-01-20 00:00 UTC (dataset start)
zeta = zeta_all[t_idx]
u = u_all[t_idx]
v = v_all[t_idx]

# Quiver downsampling stride
q_stride_x = max(1, nx // 40)
q_stride_y = max(1, ny // 40)
QX, QY = np.meshgrid(lons, lats)
u_q = u[::q_stride_y, ::q_stride_x]
v_q = v[::q_stride_y, ::q_stride_x]
qx = QX[::q_stride_y, ::q_stride_x]
qy = QY[::q_stride_y, ::q_stride_x]

plt.figure(figsize=(10, 7))
im = plt.imshow(zeta, origin=origin, extent=extent, cmap='RdBu_r', vmin=zeta_vmin, vmax=zeta_vmax, interpolation='nearest', aspect='auto')
cb = plt.colorbar(im, shrink=0.85, pad=0.02)
cb.set_label('Relative vorticity (index units)', fontsize=10)

# Quiver overlay (scaled for readability; arrows in lon/lat coords)
plt.quiver(qx, qy, u_q, v_q, color='k', angles='xy', scale_units='xy', scale=50, width=0.0025, headwidth=3)

plt.title(f'Kuroshio surface relative vorticity with velocity vectors\n{dates[t_idx]} UTC | Q={quality}, z=0', fontsize=12)
plt.xlabel('Longitude (°E)')
plt.ylabel('Latitude (°N)')
plt.xlim(lon_min, lon_max)
plt.ylim(lat_min, lat_max)

plot1_path = os.path.join(plots_dir, 'plot_1_20251201_154509.png')
plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: Okubo–Weiss parameter map with thresholded contours (same frame)
ow = ow_all[t_idx]
# Instantaneous threshold W0 = std(OW) over ocean pixels
ow_valid = ow[np.isfinite(ow)]
W0 = np.nanstd(ow_valid) if ow_valid.size > 0 else 0.0
ow_thresh = -W0

plt.figure(figsize=(10, 7))
im2 = plt.imshow(ow, origin=origin, extent=extent, cmap='RdBu_r', vmin=ow_vmin, vmax=ow_vmax, interpolation='nearest', aspect='auto')
cb2 = plt.colorbar(im2, shrink=0.85, pad=0.02)
cb2.set_label('Okubo–Weiss (index units)', fontsize=10)

# Contour highlighting rotation-dominated regions (OW < -W0)
# Build a masked array to avoid NaNs
ow_masked = np.array(ow, copy=True)
ow_masked[~np.isfinite(ow_masked)] = np.nan
# For contour, build grid coordinates
Xc, Yc = np.meshgrid(lons, lats)
# Create boolean mask
rot_mask = ow < ow_thresh
# Contour at threshold level
try:
    CS = plt.contour(Xc, Yc, ow, levels=[ow_thresh], colors='k', linewidths=1.0)
    plt.clabel(CS, inline=True, fmt={ow_thresh: f'OW = -σ ({-W0:.3f})'}, fontsize=8)
except Exception:
    pass

plt.title(f'Okubo–Weiss with eddy-core contour (OW < -σ) \n{dates[t_idx]} UTC | Q={quality}, z=0', fontsize=12)
plt.xlabel('Longitude (°E)')
plt.ylabel('Latitude (°N)')
plt.xlim(lon_min, lon_max)
plt.ylim(lat_min, lat_max)

plot2_path = os.path.join(plots_dir, 'plot_2_20251201_154509.png')
plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
plt.close()

# Plot 3: Small-multiples of daily vorticity to show evolution (every 6 days = 12 frames)
step_days = 6  # every 6 timesteps (days)
frame_indices = list(range(0, nt, step_days))[:12]  # max 12 panels
n_frames = len(frame_indices)

ncols = 4
nrows = int(np.ceil(n_frames / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows), constrained_layout=True)
axes = np.array(axes).reshape(nrows, ncols)

for ax in axes.flatten():
    ax.axis('off')

for i, (ax, ti) in enumerate(zip(axes.flatten(), frame_indices)):
    ax.imshow(zeta_all[ti], origin=origin, extent=extent, cmap='RdBu_r', vmin=zeta_vmin, vmax=zeta_vmax, interpolation='nearest', aspect='auto')
    ax.set_title(dates[ti].split(' ')[0], fontsize=9)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel('Lon (°E)', fontsize=8)
    ax.set_ylabel('Lat (°N)', fontsize=8)
    ax.tick_params(labelsize=8)
    ax.axis('on')

# Add a shared colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
norm = TwoSlopeNorm(vmin=zeta_vmin, vcenter=0.0, vmax=zeta_vmax)
sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Relative vorticity (index units)', fontsize=10)

fig.suptitle(f'Kuroshio eddy evolution (surface vorticity) — every 6 days\n2020-01-20 to 2020-03-31 | Q={quality}, z=0', fontsize=12)

plot3_path = os.path.join(plots_dir, 'plot_3_20251201_154509.png')
fig.savefig(plot3_path, dpi=150, bbox_inches='tight')
plt.close(fig)

# Plot 4: Eddy centroid tracks based on OW < -σ (simple nearest-neighbor linkage)
def indices_to_lonlat(x_idx, y_idx):
    lon = lon_min + (lon_max - lon_min) * (x_idx / (nx - 1))
    lat = lat_min + (lat_max - lat_min) * (y_idx / (ny - 1))
    return lon, lat

min_pixels = 60  # filter out tiny patches
dmax_deg = 1.0   # maximum daily jump in degrees to link a track

# Extract daily centroids
daily_centroids = []  # list of list of (lon, lat)
for ti in range(nt):
    ow_t = ow_all[ti]
    if not np.isfinite(ow_t).any():
        daily_centroids.append([])
        continue
    W0_t = np.nanstd(ow_t[np.isfinite(ow_t)])
    if not np.isfinite(W0_t) or W0_t == 0:
        daily_centroids.append([])
        continue
    mask = (ow_t < -W0_t)
    mask = np.logical_and(mask, np.isfinite(ow_t))
    if mask.sum() == 0:
        daily_centroids.append([])
        continue
    # Connected components (8-connectivity)
    labeled, num = ndimage.label(mask, structure=np.ones((3,3)))
    centroids = []
    for lbl in range(1, num+1):
        region = (labeled == lbl)
        if region.sum() < min_pixels:
            continue
        cy, cx = ndimage.center_of_mass(region)
        if np.isnan(cy) or np.isnan(cx):
            continue
        lon_c, lat_c = indices_to_lonlat(cx, cy)
        centroids.append((lon_c, lat_c))
    daily_centroids.append(centroids)

# Simple track linking
tracks = []  # list of dicts: {'lons':[], 'lats':[], 'times':[]}
active_track_ids = set()

def haversine_deg(lon1, lat1, lon2, lat2):
    # Approximate distance in degrees (not meters); using simple Pythag on degrees for short distances
    return np.sqrt((lon1 - lon2)**2 + (lat1 - lat2)**2)

for ti in range(nt):
    curr = daily_centroids[ti]
    # Build a list of active tracks (ending at ti-1)
    active = [k for k in range(len(tracks)) if len(tracks[k]['times']) > 0 and tracks[k]['times'][-1] == ti-1]
    assigned_tracks = set()
    # Try to link each current centroid to the nearest active track
    for lon_c, lat_c in curr:
        best_track = None
        best_dist = 1e9
        for k in active:
            lon_prev = tracks[k]['lons'][-1]
            lat_prev = tracks[k]['lats'][-1]
            d = haversine_deg(lon_c, lat_c, lon_prev, lat_prev)
            if d < best_dist:
                best_dist = d
                best_track = k
        if best_track is not None and best_dist <= dmax_deg and best_track not in assigned_tracks:
            # Append to this track
            tracks[best_track]['lons'].append(lon_c)
            tracks[best_track]['lats'].append(lat_c)
            tracks[best_track]['times'].append(ti)
            assigned_tracks.add(best_track)
        else:
            # Start a new track
            tracks.append({'lons': [lon_c], 'lats': [lat_c], 'times': [ti]})

# Filter tracks with at least 3 days (points)
tracks_long = [tr for tr in tracks if len(tr['times']) >= 3]

# Base: mean speed for context
speed_mean = np.nanmean(np.sqrt(u_all**2 + v_all**2), axis=0)

plt.figure(figsize=(10, 7))
im = plt.imshow(speed_mean, origin=origin, extent=extent, cmap='viridis', interpolation='nearest', aspect='auto')
cb = plt.colorbar(im, shrink=0.85, pad=0.02)
cb.set_label('Mean surface speed (arbitrary units)', fontsize=10)

# Overlay tracks
colors = plt.cm.plasma(np.linspace(0, 1, len(tracks_long))) if len(tracks_long) > 0 else []
for idx, tr in enumerate(tracks_long):
    plt.plot(tr['lons'], tr['lats'], '-', color=colors[idx], linewidth=1.8, alpha=0.9)
    plt.scatter(tr['lons'][0], tr['lats'][0], s=15, color=colors[idx], marker='o', zorder=3)
    plt.scatter(tr['lons'][-1], tr['lats'][-1], s=20, color=colors[idx], marker='s', zorder=3)

plt.title('Eddy centroid tracks (OW < -σ) over mean surface speed\n2020-01-20 to 2020-03-31 | Q={:d}, z=0'.format(quality), fontsize=12)
plt.xlabel('Longitude (°E)')
plt.ylabel('Latitude (°N)')
plt.xlim(lon_min, lon_max)
plt.ylim(lat_min, lat_max)

plot4_path = os.path.join(plots_dir, 'plot_4_20251201_154509.png')
plt.savefig(plot4_path, dpi=150, bbox_inches='tight')
plt.close()

created = 4
print(json.dumps({
    "status": "plots_created",
    "count": created,
    "paths": [plot1_path, plot2_path, plot3_path, plot4_path],
    "notes": [
        "Plot 1: surface relative vorticity with velocity quivers (origin=lower; extent set to lon/lat; Q=-6; z=0).",
        "Plot 2: Okubo–Weiss with contour at OW = -σ (rotation-dominated) highlighting eddy cores.",
        "Plot 3: Small-multiples (every 6 days) of vorticity to show eddy evolution Jan 20–Mar 31, 2020.",
        "Plot 4: Simple nearest-neighbor eddy centroid tracks (OW < -σ), over mean surface speed background."
    ],
    "metadata": {
        "lat_range": lat_range,
        "lon_range": lon_range,
        "time_range_indices": time_range,
        "timesteps_count": int(nt),
        "quality_level": quality
    }
}))