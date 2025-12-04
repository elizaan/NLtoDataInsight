import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Optional cartopy for coastlines; fall back gracefully if unavailable
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False

# Load cached data
cache_file = "/Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app/ai_data/data_cache/dyamond_llc2160/data_20251202_205719.npz"
data = np.load(cache_file, allow_pickle=True)
print("Available keys:", data.files)

# Extract arrays
u = data['u']
v = data['v']
speed = data['speed']
vorticity = data['vorticity']
sst_anom = data['sst_anom']
ocean_mask = data['ocean_mask'].astype(bool)
lat_range = data['lat_range']
lon_range = data['lon_range']
lat_vec = data['lat_vec']
lon_vec = data['lon_vec']
chosen_time_utc = str(data['chosen_time_utc'][0])
timestep_index = int(data['timestep_index'][0])
quality_level = int(data['quality_level'][0])

# Metadata
lat_min, lat_max = float(lat_range[0]), float(lat_range[1])
lon_min, lon_max = float(lon_range[0]), float(lon_range[1])
extent = [lon_min, lon_max, lat_min, lat_max]

# Output directory for plots
plots_dir = Path("/Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app/ai_data/plots/dyamond_llc2160")
plots_dir.mkdir(parents=True, exist_ok=True)

# Helpers
def finite_flat(a):
    return a[np.isfinite(a)]

def robust_limits(a, low=1, high=99):
    fa = finite_flat(a)
    if fa.size == 0:
        return (np.nan, np.nan)
    return (np.percentile(fa, low), np.percentile(fa, high))

def stats_dict(a):
    fa = finite_flat(a)
    if fa.size == 0:
        return {"min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan}
    return {
        "min": float(np.min(fa)),
        "max": float(np.max(fa)),
        "mean": float(np.mean(fa)),
        "std": float(np.std(fa))
    }

def add_percentile_lines(ax, vals, color="k", label_prefix=""):
    pcts = [1, 5, 25, 50, 75, 95, 99]
    for p in pcts:
        v = np.percentile(vals, p)
        ax.axvline(v, color=color, linestyle="--", alpha=0.5, linewidth=1)
    # Legend proxy (single label for all lines)
    ax.axvline(np.percentile(vals, 50), color=color, linestyle="--", alpha=0.5, linewidth=1, label=f"{label_prefix} percentiles")

# Plot setup utilities
def map_axes(figsize=(10, 6), title=""):
    if HAS_CARTOPY:
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.coastlines(resolution="50m", linewidth=0.6)
        ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="none", zorder=1)
        ax.gridlines(draw_labels=True, linewidth=0.2, color='gray', alpha=0.5, linestyle='--')
        ax.set_title(title, fontsize=12)
        return fig, ax
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        return fig, ax

def imshow_on_map(ax, arr, cmap, vmin=None, vmax=None, add_colorbar=True, label=""):
    if HAS_CARTOPY:
        im = ax.imshow(arr, origin="lower", extent=extent, transform=ccrs.PlateCarree(),
                       cmap=cmap, vmin=vmin, vmax=vmax, zorder=0)
    else:
        im = ax.imshow(arr, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    if add_colorbar:
        cbar = plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.9, pad=0.02)
        if label:
            cbar.set_label(label)
    return im

# 1) Speed map with sparsified quiver
speed_vmin, speed_vmax = 0.0, robust_limits(speed, 1, 99)[1]
if not np.isfinite(speed_vmax):
    speed_vmax = np.nanmax(speed)

title_common = f"Mediterranean Sea | {chosen_time_utc} UTC | Q={quality_level}\nLat [{lat_min:.2f}, {lat_max:.2f}] Lon [{lon_min:.2f}, {lon_max:.2f}]"

fig, ax = map_axes(figsize=(11, 6.5), title=f"Surface Velocity Magnitude (m/s)\n{title_common}")
imshow_on_map(ax, speed, cmap="turbo", vmin=speed_vmin, vmax=speed_vmax, label="Speed (m/s)")

# Quiver overlay (sparsify)
ny, nx = speed.shape
qstep_y = max(1, ny // 30)
qstep_x = max(1, nx // 50)
yy = lat_vec[::qstep_y]
xx = lon_vec[::qstep_x]
Uq = u[::qstep_y, ::qstep_x]
Vq = v[::qstep_y, ::qstep_x]

# Avoid plotting arrows on NaNs
mask_q = np.isfinite(Uq) & np.isfinite(Vq)
XX, YY = np.meshgrid(xx, yy)
if HAS_CARTOPY:
    ax.quiver(XX[mask_q], YY[mask_q], Uq[mask_q], Vq[mask_q], transform=ccrs.PlateCarree(),
              pivot="middle", angles='xy', scale_units='xy', scale=0.5, width=0.0015, color='k', alpha=0.6)
else:
    ax.quiver(XX[mask_q], YY[mask_q], Uq[mask_q], Vq[mask_q],
              pivot="middle", angles='xy', scale_units='xy', scale=0.5, width=0.0015, color='k', alpha=0.6)

plot1_path = plots_dir / "plot_1_20251202_205719.png"
plt.savefig(plot1_path, dpi=150, bbox_inches="tight")
plt.close(fig)

# 2) Relative vorticity map (ζ = dv/dx − du/dy)
# Use symmetric robust limits to highlight cyclonic/anticyclonic cores
vort_low, vort_high = robust_limits(vorticity, 1, 99)
vort_abs = np.nanmax(np.abs([vort_low, vort_high]))
if not np.isfinite(vort_abs) or vort_abs == 0:
    vort_abs = np.nanmax(np.abs(vorticity))
vmin_vort, vmax_vort = -vort_abs, vort_abs

fig, ax = map_axes(figsize=(11, 6.5), title=f"Surface Relative Vorticity (s⁻¹)\n{title_common}")
imshow_on_map(ax, vorticity, cmap="RdBu_r", vmin=vmin_vort, vmax=vmax_vort, label="Vorticity (s⁻¹)")
plot2_path = plots_dir / "plot_2_20251202_205719.png"
plt.savefig(plot2_path, dpi=150, bbox_inches="tight")
plt.close(fig)

# 3) SST anomaly map (relative to Mediterranean mean at this timestep)
sst_low, sst_high = robust_limits(sst_anom, 1, 99)
sst_abs = np.nanmax(np.abs([sst_low, sst_high]))
if not np.isfinite(sst_abs) or sst_abs == 0:
    sst_abs = np.nanmax(np.abs(sst_anom))
vmin_sst, vmax_sst = -sst_abs, sst_abs

fig, ax = map_axes(figsize=(11, 6.5), title=f"SST Anomaly (°C) [T - Mediterranean mean]\n{title_common}")
imshow_on_map(ax, sst_anom, cmap="RdBu_r", vmin=vmin_sst, vmax=vmax_sst, label="SST anomaly (°C)")
plot3_path = plots_dir / "plot_3_20251202_205719.png"
plt.savefig(plot3_path, dpi=150, bbox_inches="tight")
plt.close(fig)

# 4) Histograms with percentiles for Speed, Vorticity, and SST anomaly
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
# Speed
sp_vals = finite_flat(speed)
axes[0].hist(sp_vals, bins=60, color="#1f77b4", alpha=0.8)
add_percentile_lines(axes[0], sp_vals, color="#1f77b4", label_prefix="Speed")
s_stats = stats_dict(sp_vals)
axes[0].set_title("Speed (m/s)")
axes[0].set_xlabel("m/s")
axes[0].set_ylabel("Count")
axes[0].legend(loc="upper right", fontsize=8)
axes[0].text(0.02, 0.95, f"min={s_stats['min']:.3f}\nmax={s_stats['max']:.3f}\nmean={s_stats['mean']:.3f}\nstd={s_stats['std']:.3f}",
             transform=axes[0].transAxes, va="top", ha="left", fontsize=8, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

# Vorticity
vo_vals = finite_flat(vorticity)
axes[1].hist(vo_vals, bins=60, color="#d62728", alpha=0.8)
add_percentile_lines(axes[1], vo_vals, color="#d62728", label_prefix="Vort")
v_stats = stats_dict(vo_vals)
axes[1].set_title("Relative Vorticity (s⁻¹)")
axes[1].set_xlabel("s⁻¹")
axes[1].legend(loc="upper right", fontsize=8)
axes[1].text(0.02, 0.95, f"min={v_stats['min']:.2e}\nmax={v_stats['max']:.2e}\nmean={v_stats['mean']:.2e}\nstd={v_stats['std']:.2e}",
             transform=axes[1].transAxes, va="top", ha="left", fontsize=8, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

# SST anomaly
sa_vals = finite_flat(sst_anom)
axes[2].hist(sa_vals, bins=60, color="#2ca02c", alpha=0.8)
add_percentile_lines(axes[2], sa_vals, color="#2ca02c", label_prefix="SSTa")
a_stats = stats_dict(sa_vals)
axes[2].set_title("SST Anomaly (°C)")
axes[2].set_xlabel("°C")
axes[2].legend(loc="upper right", fontsize=8)
axes[2].text(0.02, 0.95, f"min={a_stats['min']:.2f}\nmax={a_stats['max']:.2f}\nmean={a_stats['mean']:.2f}\nstd={a_stats['std']:.2f}",
             transform=axes[2].transAxes, va="top", ha="left", fontsize=8, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

fig.suptitle(f"Value Distributions across Mediterranean | {chosen_time_utc} UTC", fontsize=12)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plot4_path = plots_dir / "plot_4_20251202_205719.png"
plt.savefig(plot4_path, dpi=150, bbox_inches="tight")
plt.close(fig)

# Print a concise summary for the user
print("Plots created:")
print(f"- Speed map: {plot1_path}")
print(f"- Vorticity map: {plot2_path}")
print(f"- SST anomaly map: {plot3_path}")
print(f"- Histograms: {plot4_path}")

print("Chosen timestep (UTC):", chosen_time_utc)
print("Domain bounds: lat [{:.3f}, {:.3f}], lon [{:.3f}, {:.3f}]".format(lat_min, lat_max, lon_min, lon_max))
print("Quality level:", quality_level)

NUM_PLOTS = 4
print(f"Created {NUM_PLOTS} plots successfully")