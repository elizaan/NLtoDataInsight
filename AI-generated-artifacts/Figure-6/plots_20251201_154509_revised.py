import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy import ndimage
import os
import json

# Added for map context
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D

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

# Try to extract units (if present). If not, we will explicitly state that units are as provided.
def get_unit(npz_key):
    try:
        if npz_key in data.files:
            val = data[npz_key]
            if isinstance(val, np.ndarray):
                return str(val.item()) if val.shape == () else str(val)
            return str(val)
        return None
    except Exception:
        return None

zeta_units = get_unit('zeta_units')
ow_units = get_unit('ow_units')
u_units = get_unit('u_units')
v_units = get_unit('v_units')

# Fallback labels to address unit feedback. If units are missing, label the dimensional expectation and state "as provided".
zeta_label = f"Relative vorticity ({zeta_units})" if zeta_units else "Relative vorticity (s^-1; units as provided)"
ow_label = f"Okubo–Weiss ({ow_units})" if ow_units else "Okubo–Weiss (s^-2; units as provided)"
uv_units_label = u_units if (u_units == v_units and u_units is not None) else (u_units or v_units or "units")

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

# Display smoothing (optional, light) to reduce blockiness; purely for display
SMOOTH_SIGMA = 0.8  # pixels

# Helpful note about requested vs available coverage
requested_range_txt = "Requested: 2020-01-01 to 2020-03-31"
available_start_txt = f"Available from: {dates[0].split(' ')[0]}" if len(dates) > 0 else "Available dates unknown"
coverage_note = f"{requested_range_txt} | {available_start_txt}"

# Plot 1: Surface relative vorticity with velocity quivers overlaid (first available frame)
t_idx = 0  # first available time in file
zeta = zeta_all[t_idx]
u = u_all[t_idx]
v = v_all[t_idx]

# Smooth for display
zeta_disp = ndimage.gaussian_filter(zeta, sigma=SMOOTH_SIGMA)

# Quiver downsampling stride
q_stride_x = max(1, nx // 45)  # slightly denser subsampling while reducing clutter
q_stride_y = max(1, ny // 45)
QX, QY = np.meshgrid(lons, lats)
u_q = u[::q_stride_y, ::q_stride_x]
v_q = v[::q_stride_y, ::q_stride_x]
qx = QX[::q_stride_y, ::q_stride_x]
qy = QY[::q_stride_y, ::q_stride_x]

# Choose a reference vector magnitude for the quiver key (robust 90th percentile of sampled speed)
speed_q = np.sqrt(u_q**2 + v_q**2)
ref_speed = float(np.nanpercentile(speed_q, 90)) if np.isfinite(speed_q).any() else 1.0

# Use cartopy for map context
fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
im = ax.imshow(zeta_disp, origin=origin, extent=extent, cmap='RdBu_r', vmin=zeta_vmin, vmax=zeta_vmax, interpolation='nearest', transform=ccrs.PlateCarree(), aspect='auto')
cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cb.set_label(zeta_label, fontsize=10)

# Coastlines and land
ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='lightgray', zorder=1)
ax.coastlines('50m', linewidth=0.6, zorder=2)

# Quiver overlay (arrow lengths are in display units; shown relative to lon/lat degrees)
Q = ax.quiver(qx, qy, u_q, v_q, color='k', transform=ccrs.PlateCarree(), angles='xy', scale_units='xy', scale=None, width=0.0022, headwidth=3, zorder=3)
ax.quiverkey(Q, X=0.85, Y=0.95, U=ref_speed, label=f'{ref_speed:.2f} {uv_units_label}', labelpos='E')

ax.set_title(f'Kuroshio surface relative vorticity with velocity vectors (z=0, Q={quality})\n{dates[t_idx]} UTC | {coverage_note} | display smoothing σ={SMOOTH_SIGMA}px', fontsize=12)
ax.set_xlabel('Longitude (°E)')
ax.set_ylabel('Latitude (°N)')

plot1_path = os.path.join(plots_dir, 'plot_1_20251201_154509.png')
plot1_path_revised = plot1_path.replace('.png', '_revised.png')
plt.savefig(plot1_path_revised, dpi=150, bbox_inches='tight')
plt.close(fig)

# Plot 2: Okubo–Weiss parameter map with thresholded contours (same frame)
ow = ow_all[t_idx]
ow_disp = ndimage.gaussian_filter(ow, sigma=SMOOTH_SIGMA)

# Instantaneous threshold W0 = std(OW) over ocean pixels
ow_valid = ow[np.isfinite(ow)]
W0 = np.nanstd(ow_valid) if ow_valid.size > 0 else 0.0
ow_thresh = -W0

fig2 = plt.figure(figsize=(10, 7))
ax2 = plt.axes(projection=ccrs.PlateCarree())
ax2.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
im2 = ax2.imshow(ow_disp, origin=origin, extent=extent, cmap='RdBu_r', vmin=ow_vmin, vmax=ow_vmax, interpolation='nearest', transform=ccrs.PlateCarree(), aspect='auto')
cb2 = plt.colorbar(im2, ax=ax2, shrink=0.85, pad=0.02)
cb2.set_label(ow_label, fontsize=10)

ax2.add_feature(cfeature.LAND.with_scale('50m'), facecolor='lightgray', zorder=1)
ax2.coastlines('50m', linewidth=0.6, zorder=2)

# Contour highlighting rotation-dominated regions (OW < -W0)
Xc, Yc = np.meshgrid(lons, lats)
try:
    CS = ax2.contour(Xc, Yc, ow, levels=[ow_thresh], colors='k', linewidths=1.0, transform=ccrs.PlateCarree(), zorder=3)
except Exception:
    CS = None

# Legend for OW threshold
legend_elements = [Line2D([0], [0], color='k', lw=1.0, label=f'OW < -σ (σ = std(OW) over ocean); here σ = {W0:.3e}')]
ax2.legend(handles=legend_elements, loc='lower left', fontsize=8, frameon=True)

ax2.set_title(f'Okubo–Weiss with eddy-core threshold (OW < -σ)\n{dates[t_idx]} UTC | {coverage_note} | display smoothing σ={SMOOTH_SIGMA}px', fontsize=12)
ax2.set_xlabel('Longitude (°E)')
ax2.set_ylabel('Latitude (°N)')

plot2_path = os.path.join(plots_dir, 'plot_2_20251201_154509.png')
plot2_path_revised = plot2_path.replace('.png', '_revised.png')
plt.savefig(plot2_path_revised, dpi=150, bbox_inches='tight')
plt.close(fig2)

# Plot 3: Small-multiples of daily vorticity to show evolution (1-day spacing; dense panel)
# We'll show up to 25 daily panels per page; to cover the whole available range we'll create additional pages below.
frame_indices = list(range(0, nt, 1))
per_page = 20  # dense but readable; use 4x5 grid
ncols = 5
nrows = 4
n_frames = min(len(frame_indices), per_page)

fig3, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.3*nrows), subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)
axes = np.array(axes).reshape(nrows, ncols)

# Turn off all first
for ax in axes.flatten():
    ax.set_global()
    ax.axis('off')

for i in range(n_frames):
    ax = axes.flatten()[i]
    ti = frame_indices[i]
    zeta_i = ndimage.gaussian_filter(zeta_all[ti], sigma=SMOOTH_SIGMA)
    ax.imshow(zeta_i, origin=origin, extent=extent, cmap='RdBu_r', vmin=zeta_vmin, vmax=zeta_vmax, interpolation='nearest', transform=ccrs.PlateCarree(), aspect='auto')
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='lightgray', zorder=1)
    ax.coastlines('50m', linewidth=0.4, zorder=2)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.set_title(dates[ti].split(' ')[0], fontsize=9)
    ax.axis('on')

# Shared colorbar
cbar_ax = fig3.add_axes([0.92, 0.15, 0.02, 0.7])
norm = TwoSlopeNorm(vmin=zeta_vmin, vcenter=0.0, vmax=zeta_vmax)
sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
sm.set_array([])
cbar = fig3.colorbar(sm, cax=cbar_ax)
cbar.set_label(zeta_label, fontsize=10)

fig3.suptitle(f'Kuroshio eddy evolution (surface vorticity) — daily sampling (1-day step)\n{coverage_note} | Color range fixed across dates | display smoothing σ={SMOOTH_SIGMA}px', fontsize=12)

plot3_path = os.path.join(plots_dir, 'plot_3_20251201_154509.png')
plot3_path_revised = plot3_path.replace('.png', '_revised.png')
fig3.savefig(plot3_path_revised, dpi=150, bbox_inches='tight')
plt.close(fig3)

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
speed_mean_disp = ndimage.gaussian_filter(speed_mean, sigma=SMOOTH_SIGMA)

fig4 = plt.figure(figsize=(10, 7))
ax4 = plt.axes(projection=ccrs.PlateCarree())
ax4.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
im = ax4.imshow(speed_mean_disp, origin=origin, extent=extent, cmap='viridis', interpolation='nearest', transform=ccrs.PlateCarree(), aspect='auto')
cb = plt.colorbar(im, ax=ax4, shrink=0.85, pad=0.02)
cb.set_label(f'Mean surface speed ({uv_units_label}) (as provided)', fontsize=10)

ax4.add_feature(cfeature.LAND.with_scale('50m'), facecolor='lightgray', zorder=1)
ax4.coastlines('50m', linewidth=0.6, zorder=2)

# Overlay tracks
colors = plt.cm.plasma(np.linspace(0, 1, len(tracks_long))) if len(tracks_long) > 0 else []
for idx, tr in enumerate(tracks_long):
    ax4.plot(tr['lons'], tr['lats'], '-', color=colors[idx], linewidth=1.8, alpha=0.9, transform=ccrs.PlateCarree())
    ax4.scatter(tr['lons'][0], tr['lats'][0], s=15, color=colors[idx], marker='o', zorder=3, transform=ccrs.PlateCarree())
    ax4.scatter(tr['lons'][-1], tr['lats'][-1], s=20, color=colors[idx], marker='s', zorder=3, transform=ccrs.PlateCarree())

ax4.set_title('Eddy centroid tracks (OW < -σ) over mean surface speed\n{} | Q={:d}, z=0 | {}'.format(coverage_note, quality, f"display smoothing σ={SMOOTH_SIGMA}px"), fontsize=12)
ax4.set_xlabel('Longitude (°E)')
ax4.set_ylabel('Latitude (°N)')

plot4_path = os.path.join(plots_dir, 'plot_4_20251201_154509.png')
plot4_path_revised = plot4_path.replace('.png', '_revised.png')
plt.savefig(plot4_path_revised, dpi=150, bbox_inches='tight')
plt.close(fig4)

# NEW: Daily small-multiples across the entire available range (paginate with 25 frames per page) to show 1-day sampling explicitly
# Highest existing plot number is 4, so new plots start at 5 with "_revised" suffix.
daily_indices = list(range(nt))
per_page_new = 25
n_pages = int(np.ceil(len(daily_indices) / per_page_new))
ncols_new = 5
nrows_new = 5

new_plot_paths = []
for page in range(n_pages):
    start = page * per_page_new
    end = min((page + 1) * per_page_new, len(daily_indices))
    idxs = daily_indices[start:end]

    figN, axesN = plt.subplots(nrows_new, ncols_new, figsize=(3.8*ncols_new, 3.0*nrows_new), subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)
    axesN = np.array(axesN).reshape(nrows_new, ncols_new)
    for ax in axesN.flatten():
        ax.set_global()
        ax.axis('off')

    for i, ti in enumerate(idxs):
        ax = axesN.flatten()[i]
        zeta_i = ndimage.gaussian_filter(zeta_all[ti], sigma=SMOOTH_SIGMA)
        ax.imshow(zeta_i, origin=origin, extent=extent, cmap='RdBu_r', vmin=zeta_vmin, vmax=zeta_vmax, interpolation='nearest', transform=ccrs.PlateCarree(), aspect='auto')
        ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='lightgray', zorder=1)
        ax.coastlines('50m', linewidth=0.3, zorder=2)
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.set_title(dates[ti].split(' ')[0], fontsize=8)
        ax.axis('on')

    # Shared colorbar
    cbar_axN = figN.add_axes([0.92, 0.15, 0.02, 0.7])
    normN = TwoSlopeNorm(vmin=zeta_vmin, vcenter=0.0, vmax=zeta_vmax)
    smN = plt.cm.ScalarMappable(cmap='RdBu_r', norm=normN)
    smN.set_array([])
    cbarN = figN.colorbar(smN, cax=cbar_axN)
    cbarN.set_label(zeta_label, fontsize=10)

    figN.suptitle(f'Daily surface vorticity — page {page+1}/{n_pages} (1-day step)\n{coverage_note} | Color range fixed across dates | display smoothing σ={SMOOTH_SIGMA}px', fontsize=12)

    new_plot_num = 5 + page  # start at 5
    new_path = os.path.join(plots_dir, f'plot_{new_plot_num}_20251201_154509_revised.png')
    figN.savefig(new_path, dpi=150, bbox_inches='tight')
    plt.close(figN)
    new_plot_paths.append(new_path)

# JSON summary
created = 4 + len(new_plot_paths)
all_paths = [plot1_path_revised, plot2_path_revised, plot3_path_revised, plot4_path_revised] + new_plot_paths
print(json.dumps({
    "status": "plots_created",
    "count": created,
    "paths": all_paths,
    "notes": [
        "Plot 1: Added coastlines/projection (PlateCarree), daily frame with quiver key, and display smoothing; fixed color range across dates; units labeled explicitly.",
        "Plot 2: Okubo–Weiss with eddy-core contour; legend specifies OW threshold as OW < -σ with σ computed over ocean pixels; map context added; consistent color limits.",
        "Plot 3: Small-multiples now use daily sampling (1-day step) with map context; fixed colorbar across dates; display smoothing; caption clarifies requested vs. available coverage.",
        "Plot 4: Eddy centroid tracks plotted on a mean-speed background with coastlines; caption clarifies requested vs. available coverage; display smoothing for background.",
        "Plots 5..N: Paginated daily small-multiples to cover the full available time range at 1-day spacing."
    ],
    "metadata": {
        "lat_range": lat_range,
        "lon_range": lon_range,
        "time_range_indices": time_range,
        "timesteps_count": int(nt),
        "quality_level": quality,
        "units": {
            "zeta": zeta_units or "not provided (shown as s^-1; as provided)",
            "okubo_weiss": ow_units or "not provided (shown as s^-2; as provided)",
            "u": u_units or "not provided",
            "v": v_units or "not provided"
        }
    }
}))