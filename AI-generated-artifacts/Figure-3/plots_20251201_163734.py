import numpy as np
import matplotlib.pyplot as plt
import os
import json

# Try to use cartopy for coastlines/map; fall back to plain axes if unavailable
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False

# Paths
cache_path = "/Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app/ai_data/data_cache/dyamond_llc2160/data_20251201_163734.npz"
plots_dir = "/Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app/ai_data/plots/dyamond_llc2160"
os.makedirs(plots_dir, exist_ok=True)

# Load cached data
data = np.load(cache_path, allow_pickle=True)
print("Available keys:", data.files)

# Extract arrays and metadata
S = data["salinity"]  # (ny, nx), NaN on land
grad_mag = data["grad_mag_psu_per_km"]  # (ny, nx)
dSdx = data["dSdx"]  # PSU/m
dSdy = data["dSdy"]
ocean_mask = data["ocean_mask"].astype(bool)
lon_vec = data["lon_vec"]
lat_vec = data["lat_vec"]
lon_min, lon_max = float(data["lon_range"][0]), float(data["lon_range"][1])
lat_min, lat_max = float(data["lat_range"][0]), float(data["lat_range"][1])
quality = int(data["quality_level"])
time_iso = str(data["time_iso"][0])

ny, nx = S.shape
# Build coordinate grids
lon2d, lat2d = np.meshgrid(lon_vec, lat_vec)

# Plot settings
p95 = float(np.nanpercentile(grad_mag, 95))
vmin, vmax = 0.0, max(p95, 1e-3)  # avoid too tiny vmax
cmap_mag = "viridis"

# 1) Map: |∇S| (PSU/km) with coastline
figsize = (10, 8)
if HAS_CARTOPY:
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=proj)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
    im = ax.imshow(
        grad_mag,
        origin="lower",
        extent=[lon_min, lon_max, lat_min, lat_max],
        transform=proj,
        cmap=cmap_mag,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest"
    )
    ax.coastlines(resolution="110m", linewidth=0.8)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='k', alpha=0.3, linestyle='--')
    cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02)
else:
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        grad_mag,
        origin="lower",
        extent=[lon_min, lon_max, lat_min, lat_max],
        cmap=cmap_mag,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="auto"
    )
    cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02)

cbar.set_label("|∇S| (PSU/km)")
ax.set_title(f"Bay of Bengal near-surface salinity gradient |∇S|\n{time_iso} (Q={quality})")
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
out1 = os.path.join(plots_dir, "plot_1_20251201_163734.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig)

# 2) Salinity contours + sparse gradient direction vectors
# Prepare decimated grid
step = 15  # decimation factor
Yidx = np.arange(0, ny, step)
Xidx = np.arange(0, nx, step)
Lon_q = lon2d[Yidx][:, Xidx]
Lat_q = lat2d[Yidx][:, Xidx]
dSdx_q = dSdx[Yidx][:, Xidx]
dSdy_q = dSdy[Yidx][:, Xidx]
mask_q = ocean_mask[Yidx][:, Xidx]
mag_q = np.sqrt(dSdx_q**2 + dSdy_q**2)

# Unit vectors for direction only; scale to a fixed small arrow length in degrees
eps = 1e-20
udir = np.where(mag_q > 0, dSdx_q / (mag_q + eps), 0.0)
vdir = np.where(mag_q > 0, dSdy_q / (mag_q + eps), 0.0)
arrow_deg = 0.25  # arrow length in degrees (roughly uniform)
U_plot = udir * arrow_deg
V_plot = vdir * arrow_deg

# Mask out land/NaN
U_plot = np.where(mask_q & np.isfinite(mag_q), U_plot, np.nan)
V_plot = np.where(mask_q & np.isfinite(mag_q), V_plot, np.nan)

# Contour levels for salinity (robust range)
S_min = float(np.nanpercentile(S, 5))
S_max = float(np.nanpercentile(S, 95))
nlevels = 10
levels = np.linspace(S_min, S_max, nlevels)

if HAS_CARTOPY:
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=proj)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
    # Salinity contours
    cs = ax.contour(
        lon2d, lat2d, S,
        levels=levels,
        colors="k",
        linewidths=0.6,
        transform=proj
    )
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")
    # Quiver for gradient directions (toward increasing salinity)
    # Note: quiver U,V are in data units (degrees here)
    q = ax.quiver(
        Lon_q, Lat_q, U_plot, V_plot,
        transform=proj,
        color="crimson",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.002,
        headwidth=3,
        headlength=4
    )
    ax.coastlines(resolution="110m", linewidth=0.8)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='k', alpha=0.3, linestyle='--')
else:
    fig, ax = plt.subplots(figsize=(10, 8))
    cs = ax.contour(
        lon2d, lat2d, S,
        levels=levels,
        colors="k",
        linewidths=0.6
    )
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")
    q = ax.quiver(
        Lon_q, Lat_q, U_plot, V_plot,
        color="crimson",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.002,
        headwidth=3,
        headlength=4
    )
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

ax.set_title(f"Salinity isohalines and gradient directions (surface)\n{time_iso} (arrows: toward increasing S; fixed length)")
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
out2 = os.path.join(plots_dir, "plot_2_20251201_163734.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig)

# 3) Histogram of |∇S| (PSU/km), log-scaled x-axis
grad_vals = grad_mag[np.isfinite(grad_mag) & ocean_mask]
if grad_vals.size > 0:
    gmin = max(1e-5, float(np.nanmin(grad_vals)))
    g95 = float(np.nanpercentile(grad_vals, 95))
    if g95 <= gmin:
        g95 = gmin * 10.0
    bins = np.logspace(np.log10(gmin), np.log10(g95), 40)
else:
    bins = np.logspace(-5, -1, 40)

fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(grad_vals, bins=bins, color="steelblue", alpha=0.85, edgecolor="white")
ax.set_xscale("log")
ax.set_xlabel("|∇S| (PSU/km) [log scale]")
ax.set_ylabel("Count")
ax.set_title(f"Distribution of near-surface salinity gradient magnitudes\n{time_iso}")
# Mark key percentiles
if grad_vals.size > 0:
    p50 = float(np.nanpercentile(grad_vals, 50))
    p90 = float(np.nanpercentile(grad_vals, 90))
    ax.axvline(p50, color="red", linestyle="--", linewidth=1, label=f"Median: {p50:.3f}")
    ax.axvline(p90, color="orange", linestyle="--", linewidth=1, label=f"90th pct: {p90:.3f}")
    ax.legend(frameon=False, fontsize=9)
out3 = os.path.join(plots_dir, "plot_3_20251201_163734.png")
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close(fig)

print(json.dumps({
    "created_plots": [out1, out2, out3],
    "cartopy_used": HAS_CARTOPY,
    "timestamp": time_iso,
    "quality": quality,
    "region": {
        "lon_range": [lon_min, lon_max],
        "lat_range": [lat_min, lat_max]
    },
    "notes": "Map uses origin=lower with explicit extent; gradient vectors are unit-direction arrows of fixed length to indicate direction of increasing salinity."
}))