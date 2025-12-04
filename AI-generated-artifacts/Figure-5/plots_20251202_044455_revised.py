import os
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Paths
cache_path = "/Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app/ai_data/data_cache/dyamond_llc2160/data_20251202_044455.npz"
plots_dir = Path("/Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app/ai_data/plots/dyamond_llc2160")
plots_dir.mkdir(parents=True, exist_ok=True)

# Load cached data
data = np.load(cache_path, allow_pickle=True)
print("Available keys:", data.files)

# Extract static metadata
lat_range = data["lat_range"].tolist()
lon_range = data["lon_range"].tolist()
x_range = data["x_range"].tolist()
y_range = data["y_range"].tolist()
quality = int(data["quality_level"][0]) if "quality_level" in data else -3
stride_hours = int(data["stride_hours"][0]) if "stride_hours" in data else 24
months = [str(m) for m in data["months"]]

lat_min, lat_max = float(lat_range[0]), float(lat_range[1])
lon_min, lon_max = float(lon_range[0]), float(lon_range[1])

# Helper to compute gradient magnitude safely
def grad_mag(field2d):
    # Fill NaNs with overall mean to make gradients well-defined
    if np.isnan(field2d).any():
        fill_val = float(np.nanmean(field2d))
        fld = np.nan_to_num(field2d, nan=fill_val)
    else:
        fld = field2d
    gy, gx = np.gradient(fld)
    return np.sqrt(gx * gx + gy * gy)

# Helper: longitude tick formatter with E/W suffix
def _lon_fmt(val, pos=None):
    if not np.isfinite(val):
        return ""
    if val == 0:
        return "0°"
    hemi = "E" if val > 0 else "W"
    return f"{abs(val):.0f}°{hemi}"

def apply_lon_formatter(ax, label="Longitude (°E/°W)"):
    ax.xaxis.set_major_formatter(FuncFormatter(_lon_fmt))
    ax.set_xlabel(label)

# Helper: create "_revised" filename next to original
def revised_path(p: Path) -> Path:
    return Path(str(p).replace(".png", "_revised.png"))

# Helper: build a front-following path (option c in feedback)
def build_front_following_path(sst_month_mean, lat_coords, lon_coords, lat_window=(36.0, 43.0)):
    mag = grad_mag(sst_month_mean)
    ny, nx = mag.shape
    # Restrict to latitude window if it intersects domain; else use full domain
    if (lat_coords.min() <= lat_window[1]) and (lat_coords.max() >= lat_window[0]):
        win_mask = (lat_coords >= max(lat_window[0], lat_coords.min())) & (lat_coords <= min(lat_window[1], lat_coords.max()))
    else:
        win_mask = np.ones_like(lat_coords, dtype=bool)

    mag_win = mag.copy()
    mag_win[~win_mask, :] = np.nan

    y_idx_path = np.empty(nx, dtype=int)
    for j in range(nx):
        col = mag_win[:, j]
        if np.all(~np.isfinite(col)):  # fallback if window produced all-NaN
            col = mag[:, j]
        if np.all(~np.isfinite(col)):
            y_idx_path[j] = ny // 2
        else:
            y_idx_path[j] = int(np.nanargmax(col))
    lat_path = lat_coords[y_idx_path]

    # Smooth the path lightly to reduce pixel-to-pixel zig-zag (simple moving average)
    k = 7
    kernel = np.ones(k) / k
    lat_path_smooth = np.convolve(lat_path, kernel, mode="same")

    return y_idx_path, lat_path_smooth, mag

# Iterate over requested months and plot
plot_index = 1
total_plots = 0

for m in months:
    # Required keys for this month
    k_sst_daily = f"{m}_sst_daily"
    k_sst_mean = f"{m}_sst_month_mean"
    k_transect_anom = f"{m}_transect_anom"
    k_timesteps = f"{m}_timesteps"
    k_times_iso = f"{m}_times_iso"
    k_dom_mean_sst = f"{m}_dom_mean_sst"
    k_front_index = f"{m}_front_index"
    k_dom_std_anom = f"{m}_dom_std_anom"
    k_transect_y_index = f"{m}_transect_y_index"

    # Check availability
    required = [k_sst_daily, k_sst_mean, k_transect_anom, k_timesteps, k_times_iso, k_dom_mean_sst, k_front_index, k_dom_std_anom, k_transect_y_index]
    missing = [k for k in required if k not in data.files]
    if len(missing) > 0:
        print(json.dumps({"warning": f"Skipping month {m} because missing keys: {missing}"}))
        continue

    # Load arrays
    sst_daily = data[k_sst_daily]            # (nt, ny, nx)
    sst_month_mean = data[k_sst_mean]        # (ny, nx)
    transect_anom = data[k_transect_anom]    # (nt, nx)  # not used below; replaced by front-following transect
    timesteps = data[k_timesteps]            # (nt,)
    times_iso = [str(t) for t in data[k_times_iso]]
    dom_mean_sst = data[k_dom_mean_sst]      # (nt,)
    front_index = data[k_front_index]        # (nt,)
    dom_std_anom = data[k_dom_std_anom]      # (nt,)
    transect_y_index = int(data[k_transect_y_index][0])

    nt, ny, nx = sst_daily.shape
    # Coordinates for imshow extents
    extent = [lon_min, lon_max, lat_min, lat_max]
    lon_coords = np.linspace(lon_min, lon_max, nx)
    lat_coords = np.linspace(lat_min, lat_max, ny)

    # Build a front-following path (automatic option c)
    y_idx_path, lat_path, mag_month = build_front_following_path(sst_month_mean, lat_coords, lon_coords)

    # Recompute transect (lon-time) anomaly along the front-following path
    x_idx = np.arange(nx)
    sst_month_path = sst_month_mean[y_idx_path, x_idx]  # (nx,)
    transect_series = np.empty((nt, nx), dtype=float)
    for k in range(nt):
        transect_series[k, :] = sst_daily[k, y_idx_path, x_idx]
    transect_anom_ff = transect_series - sst_month_path[None, :]  # (nt, nx)
    transect_anom_ff = np.where(np.isfinite(transect_anom_ff), transect_anom_ff, np.nan)

    # Path-mean time series diagnostics
    sst_path_mean = np.nanmean(transect_series, axis=1)              # mean along path
    sst_month_path_mean = float(np.nanmean(sst_month_path))          # scalar ref for anomaly definition below
    path_mean_anom = sst_path_mean - sst_month_path_mean             # anomaly relative to month-mean along-path average

    # Front intensity time series averaged along the front-following path
    front_index_path = np.empty(nt, dtype=float)
    for k in range(nt):
        gk = grad_mag(sst_daily[k])
        front_index_path[k] = float(np.nanmean(gk[y_idx_path, x_idx]))

    # Domain-mean anomaly time series (relative to month-mean domain average)
    sst_month_mean_dom_mean = float(np.nanmean(sst_month_mean))
    dom_mean_anom = dom_mean_sst - sst_month_mean_dom_mean

    # Compute a global anomaly range for consistent color scale (daily anomalies)
    sst_anom_all = sst_daily - sst_month_mean[None, :, :]
    q_lo = np.nanpercentile(sst_anom_all, 1)
    q_hi = np.nanpercentile(sst_anom_all, 99)
    vlim = max(abs(q_lo), abs(q_hi))
    if not np.isfinite(vlim) or vlim == 0:
        vlim = 2.0  # fallback

    # ------------- Plot 1: Month-mean SST with |∇T| contours and front-following path + inset -------------
    plt.figure(figsize=(10, 7))
    im = plt.imshow(sst_month_mean, origin='lower', extent=extent, cmap='turbo')
    cb = plt.colorbar(im, shrink=0.85, pad=0.02)
    cb.set_label("SST month-mean (°C)")

    # Thinned/softened gradient contours from month mean
    finite_mag = mag_month[np.isfinite(mag_month)]
    if finite_mag.size > 0:
        levels = np.quantile(finite_mag, [0.80, 0.90, 0.97])
        cs = plt.contour(lon_coords, lat_coords, mag_month, levels=levels, colors='k', linewidths=0.4, alpha=0.5)

    # Front-following path
    plt.plot(lon_coords, lat_path, color='white', lw=1.8, alpha=0.95, label="Front-following path (max |∇T| per longitude)")

    # Inset to show the path clearly (zoom to path latitude band)
    ax_ins = inset_axes(plt.gca(), width="37%", height="37%", loc="upper right", borderpad=1.2)
    ax_ins.imshow(sst_month_mean, origin='lower', extent=extent, cmap='turbo')
    ax_ins.plot(lon_coords, lat_path, color='white', lw=2.0, alpha=1.0)
    # Zoom vertically around the path band
    pad_lat = max(1.0, 0.5 * np.nanstd(lat_path))
    y0, y1 = np.nanmin(lat_path) - pad_lat, np.nanmax(lat_path) + pad_lat
    ax_ins.set_ylim([y0, y1])
    ax_ins.set_xlim([lon_min, lon_max])
    ax_ins.set_xticks([])
    ax_ins.set_yticks([])
    ax_ins.set_title("Path (zoom)", fontsize=9)

    plt.legend(loc='lower right', frameon=True)
    plt.title(f"{m} Gulf Stream: Month-mean SST with front contours |∇T|\nQ={quality}, daily stride={stride_hours}h")
    apply_lon_formatter(plt.gca(), "Longitude (°E/°W)")
    plt.ylabel("Latitude (°N)")
    out1 = plots_dir / f"plot_{plot_index}_20251202_044455.png"
    plt.savefig(revised_path(out1), dpi=150, bbox_inches='tight')
    plt.close()
    plot_index += 1
    total_plots += 1

    # ------------- Plot 2: Three daily anomaly snapshots with |∇T| contours; standardized anomaly definition -------------
    sel_idxs = [0, nt // 2, nt - 1]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    im_for_cbar = None
    for ax, k in zip(axes, sel_idxs):
        anom_k = sst_daily[k] - sst_month_mean
        im2 = ax.imshow(anom_k, origin='lower', extent=extent, cmap='RdBu_r', vmin=-vlim, vmax=vlim)
        if im_for_cbar is None:
            im_for_cbar = im2
        # Gradient magnitude from the day's SST (not anomaly), softened
        mag_k = grad_mag(sst_daily[k])
        finite_mag_k = mag_k[np.isfinite(mag_k)]
        if finite_mag_k.size > 0:
            lev_k = np.quantile(finite_mag_k, [0.85, 0.95, 0.99])
            ax.contour(lon_coords, lat_coords, mag_k, levels=lev_k, colors='k', linewidths=0.4, alpha=0.45)
        # Overlay front-following path
        ax.plot(lon_coords, lat_path, color='k', linestyle='--', linewidth=1.0)
        # Subtitle with date and anomaly definition
        date_label = times_iso[k].replace("T", " ")
        ax.set_title(f"SST anomaly (°C) relative to the {m} monthly mean\n{date_label}")
        apply_lon_formatter(ax, "Longitude (°E/°W)")
        ax.set_ylabel("Latitude (°N)")
    # Shared symmetric colorbar
    cax = fig.add_axes([0.92, 0.13, 0.015, 0.74])
    cbar = fig.colorbar(im_for_cbar, cax=cax)
    cbar.set_ticks([-(vlim), 0, vlim])
    cbar.set_ticklabels([f"{-vlim:.1f}", "0", f"{vlim:.1f}"])
    cbar.set_label("SST anomaly (°C)")
    fig.suptitle(f"{m} Gulf Stream: Daily SST anomalies (3 snapshots) with front contours |∇T|\nQ={quality}, daily stride={stride_hours}h", y=1.02)
    out2 = plots_dir / f"plot_{plot_index}_20251202_044455.png"
    fig.savefig(revised_path(out2), dpi=150, bbox_inches='tight')
    plt.close(fig)
    plot_index += 1
    total_plots += 1

    # ------------- Plot 3: Hovmöller (lon-time) of transect anomaly along front-following path; mask invalid -------------
    # Prepare masked array to remove land/invalid (gray)
    transect_ma = np.ma.masked_invalid(transect_anom_ff)
    # Robust symmetric color limits
    q_lo_t = np.nanpercentile(transect_ma, 1)
    q_hi_t = np.nanpercentile(transect_ma, 99)
    vlim_t = max(abs(q_lo_t), abs(q_hi_t))
    if not np.isfinite(vlim_t) or vlim_t == 0:
        vlim_t = vlim

    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()
    # Use a colormap with explicit "bad" color for masked regions
    cmap_hov = plt.cm.get_cmap('RdBu_r').copy()
    cmap_hov.set_bad(color='lightgray')  # land/invalid
    im3 = ax.imshow(transect_ma, origin='lower', aspect='auto',
                    extent=[lon_min, lon_max, 0, nt - 1],
                    cmap=cmap_hov, vmin=-vlim_t, vmax=vlim_t)
    # Y ticks every ~5 segments
    step = max(1, nt // 6)
    yticks = list(range(0, nt, step))
    ax.set_yticks(yticks)
    # Convert times_iso to labels
    time_labels = []
    for idx in yticks:
        try:
            dt = datetime.fromisoformat(times_iso[idx])
            time_labels.append(dt.strftime("%m-%d"))
        except Exception:
            time_labels.append(str(idx))
    ax.set_yticklabels(time_labels)
    cb3 = plt.colorbar(im3, pad=0.02)
    cb3.set_label("SST anomaly (°C) at front-following path")
    apply_lon_formatter(ax, "Longitude (°E/°W)")
    ax.set_ylabel("Date (MM-DD)")
    ax.set_title(f"{m} Hovmöller (lon–time) of SST anomaly\nFront-following path of max |∇T|; Q={quality}, stride={stride_hours}h")
    out3 = plots_dir / f"plot_{plot_index}_20251202_044455.png"
    plt.savefig(revised_path(out3), dpi=150, bbox_inches='tight')
    plt.close()
    plot_index += 1
    total_plots += 1

    # ------------- Plot 4: Time series along the front-following path -------------
    # Convert times for x-axis
    times_dt = []
    for t in times_iso:
        try:
            times_dt.append(datetime.fromisoformat(t))
        except Exception:
            times_dt.append(None)
    # Filter out None if needed (fallback: index)
    x_vals = np.arange(len(times_dt))
    use_datetime_axis = all([td is not None for td in times_dt])

    fig, ax1 = plt.subplots(figsize=(12, 5))
    if use_datetime_axis:
        ax1.plot(times_dt, path_mean_anom, 'b-o', ms=3, lw=1.2, label="Path-mean SST anomaly")
        x_label = "Date"
    else:
        ax1.plot(x_vals, path_mean_anom, 'b-o', ms=3, lw=1.2, label="Path-mean SST anomaly")
        x_label = "Day index"
    ax1.set_ylabel("Anomaly (°C) relative to the month’s along-path mean", color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    if use_datetime_axis:
        ax2.plot(times_dt, front_index_path, 'r-s', ms=3, lw=1.2, label="Front-intensity |∇T| (path mean)")
    else:
        ax2.plot(x_vals, front_index_path, 'r-s', ms=3, lw=1.2, label="Front-intensity |∇T| (path mean)")
    ax2.set_ylabel("|∇T| (°C per grid unit)", color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True)
    ax1.set_xlabel(x_label)
    plt.title(f"{m} Front-following path: SST anomaly and front-intensity index\nQ={quality}, daily stride={stride_hours}h")
    out4 = plots_dir / f"plot_{plot_index}_20251202_044455.png"
    plt.savefig(revised_path(out4), dpi=150, bbox_inches='tight')
    plt.close()
    plot_index += 1
    total_plots += 1

    # ------------- Plot 5: Month-mean front-intensity map (mean |∇T| over daily fields) + path overlay -------------
    # Compute average |∇T| across days
    front_stack = np.zeros((nt, ny, nx), dtype=np.float32)
    for k in range(nt):
        front_stack[k] = grad_mag(sst_daily[k])
    front_mean = np.nanmean(front_stack, axis=0)

    plt.figure(figsize=(10, 7))
    im5 = plt.imshow(front_mean, origin='lower', extent=extent, cmap='magma')
    cb5 = plt.colorbar(im5, shrink=0.85, pad=0.02)
    cb5.set_label("Mean |∇T| (°C per grid unit) over month")
    # Overlay front-following path
    plt.plot(lon_coords, lat_path, color='white', linestyle='--', linewidth=1.4, alpha=0.95)
    plt.title(f"{m} Gulf Stream: Month-mean front intensity map (|∇T|)\nQ={quality}, daily stride={stride_hours}h")
    apply_lon_formatter(plt.gca(), "Longitude (°E/°W)")
    plt.ylabel("Latitude (°N)")
    out5 = plots_dir / f"plot_{plot_index}_20251202_044455.png"
    plt.savefig(revised_path(out5), dpi=150, bbox_inches='tight')
    plt.close()
    plot_index += 1
    total_plots += 1

    # ------------- NEW Plot 6 (added): Month-mean spatial anomaly map (relative to month’s domain-mean) -------------
    month_spatial_anom = sst_month_mean - sst_month_mean_dom_mean
    q_lo_m = np.nanpercentile(month_spatial_anom, 1)
    q_hi_m = np.nanpercentile(month_spatial_anom, 99)
    vlim_m = max(abs(q_lo_m), abs(q_hi_m))
    if not np.isfinite(vlim_m) or vlim_m == 0:
        vlim_m = 2.0

    plt.figure(figsize=(10, 7))
    im6 = plt.imshow(month_spatial_anom, origin='lower', extent=extent, cmap='RdBu_r', vmin=-vlim_m, vmax=vlim_m)
    cb6 = plt.colorbar(im6, shrink=0.85, pad=0.02)
    cb6.set_label("SST spatial anomaly (°C)")
    # Overlay front-following path
    plt.plot(lon_coords, lat_path, color='k', linestyle='--', linewidth=1.0)
    plt.title(f"{m} SST spatial anomaly map (°C) relative to the {m} month’s domain-mean\nQ={quality}, daily stride={stride_hours}h")
    apply_lon_formatter(plt.gca(), "Longitude (°E/°W)")
    plt.ylabel("Latitude (°N)")
    out6 = plots_dir / f"plot_{plot_index}_20251202_044455.png"
    plt.savefig(revised_path(out6), dpi=150, bbox_inches='tight')
    plt.close()
    plot_index += 1
    total_plots += 1

print(f"Created {total_plots} plots successfully")