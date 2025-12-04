import os
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

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
    transect_anom = data[k_transect_anom]    # (nt, nx)
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
    transect_lat = float(lat_coords[transect_y_index])

    # Domain-mean anomaly time series (relative to month-mean domain average)
    sst_month_mean_dom_mean = float(np.nanmean(sst_month_mean))
    dom_mean_anom = dom_mean_sst - sst_month_mean_dom_mean

    # Compute a global anomaly range for consistent color scale
    sst_anom_all = sst_daily - sst_month_mean[None, :, :]
    # Robust symmetric vlim
    q_lo = np.nanpercentile(sst_anom_all, 1)
    q_hi = np.nanpercentile(sst_anom_all, 99)
    vlim = max(abs(q_lo), abs(q_hi))
    if not np.isfinite(vlim) or vlim == 0:
        vlim = 2.0  # fallback

    # Plot 1: Month-mean SST with |∇T| contours and transect line
    plt.figure(figsize=(10, 7))
    im = plt.imshow(sst_month_mean, origin='lower', extent=extent, cmap='turbo')
    cb = plt.colorbar(im, shrink=0.85, pad=0.02)
    cb.set_label("SST month-mean (°C)")
    mag = grad_mag(sst_month_mean)
    # Choose contour levels from upper quantiles to highlight strong fronts
    finite_mag = mag[np.isfinite(mag)]
    if finite_mag.size > 0:
        levels = np.quantile(finite_mag, [0.70, 0.85, 0.95])
        plt.contour(lon_coords, lat_coords, mag, levels=levels, colors='k', linewidths=0.6, alpha=0.8)
    # Transect line
    plt.hlines(transect_lat, lon_min, lon_max, colors='white', linestyles='--', linewidth=1.0, label=f"Transect at ~{transect_lat:.2f}°N")
    plt.legend(loc='lower right', frameon=True)
    plt.title(f"{m} Gulf Stream: Month-mean SST with front contours |∇T|\nQ={quality}, daily stride={stride_hours}h")
    plt.xlabel("Longitude (°E)")
    plt.ylabel("Latitude (°N)")
    out1 = plots_dir / f"plot_{plot_index}_20251202_044455.png"
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close()
    plot_index += 1
    total_plots += 1

    # Plot 2: Three daily anomaly snapshots with |∇T| contours
    sel_idxs = [0, nt // 2, nt - 1]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, k in zip(axes, sel_idxs):
        anom_k = sst_daily[k] - sst_month_mean
        im2 = ax.imshow(anom_k, origin='lower', extent=extent, cmap='RdBu_r', vmin=-vlim, vmax=vlim)
        # Gradient magnitude from the day's SST (not anomaly)
        mag_k = grad_mag(sst_daily[k])
        finite_mag_k = mag_k[np.isfinite(mag_k)]
        if finite_mag_k.size > 0:
            lev_k = np.quantile(finite_mag_k, [0.80, 0.90, 0.98])
            ax.contour(lon_coords, lat_coords, mag_k, levels=lev_k, colors='k', linewidths=0.5, alpha=0.8)
        # Overlay transect
        ax.hlines(transect_lat, lon_min, lon_max, colors='k', linestyles='--', linewidth=0.8)
        # Subtitle with date
        date_label = times_iso[k].replace("T", " ")
        ax.set_title(f"Anomaly + |∇T|: {date_label}")
        ax.set_xlabel("Lon (°E)")
        ax.set_ylabel("Lat (°N)")
    cax = fig.add_axes([0.92, 0.13, 0.015, 0.74])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='RdBu_r'), cax=cax)
    # Manually set colorbar ticks to match vmin/vmax
    cbar.set_ticks([-(vlim), 0, vlim])
    cbar.set_ticklabels([f"{-vlim:.1f}", "0", f"{vlim:.1f}"])
    cbar.set_label("SST anomaly (°C)")
    fig.suptitle(f"{m} Gulf Stream: Daily SST anomalies (3 snapshots) with front contours |∇T|\nQ={quality}, daily stride={stride_hours}h", y=1.02)
    out2 = plots_dir / f"plot_{plot_index}_20251202_044455.png"
    fig.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    plot_index += 1
    total_plots += 1

    # Plot 3: Hovmöller (lon-time) of transect anomaly
    # Build y-axis time as index; we will label with dates
    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()
    # Use imshow with extent mapping time index [0, nt-1] to y dimension
    im3 = ax.imshow(transect_anom, origin='lower', aspect='auto',
                    extent=[lon_min, lon_max, 0, nt - 1], cmap='RdBu_r')
    # Set symmetric color limits
    q_lo_t = np.nanpercentile(transect_anom, 1)
    q_hi_t = np.nanpercentile(transect_anom, 99)
    vlim_t = max(abs(q_lo_t), abs(q_hi_t))
    if not np.isfinite(vlim_t) or vlim_t == 0:
        vlim_t = vlim
    im3.set_clim(-vlim_t, vlim_t)
    # Y ticks every ~5 days
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
    cb3.set_label("SST anomaly (°C) at transect")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Date (MM-DD)")
    ax.set_title(f"{m} Hovmöller (lon–time) of SST anomaly\nTransect at ~{transect_lat:.2f}°N; Q={quality}, stride={stride_hours}h")
    out3 = plots_dir / f"plot_{plot_index}_20251202_044455.png"
    plt.savefig(out3, dpi=150, bbox_inches='tight')
    plt.close()
    plot_index += 1
    total_plots += 1

    # Plot 4: Time series of domain-mean anomaly and front-intensity index
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
        ax1.plot(times_dt, dom_mean_anom, 'b-o', ms=3, lw=1.2, label="Domain-mean SST anomaly")
        x_label = "Date"
    else:
        ax1.plot(x_vals, dom_mean_anom, 'b-o', ms=3, lw=1.2, label="Domain-mean SST anomaly")
        x_label = "Day index"
    ax1.set_ylabel("Anomaly (°C)", color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    if use_datetime_axis:
        ax2.plot(times_dt, front_index, 'r-s', ms=3, lw=1.2, label="Front-intensity |∇T| (mean)")
    else:
        ax2.plot(x_vals, front_index, 'r-s', ms=3, lw=1.2, label="Front-intensity |∇T| (mean)")
    ax2.set_ylabel("|∇T| (°C per grid unit)", color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True)
    ax1.set_xlabel(x_label)
    plt.title(f"{m} Domain-mean SST anomaly and front-intensity index\nQ={quality}, daily stride={stride_hours}h")
    out4 = plots_dir / f"plot_{plot_index}_20251202_044455.png"
    plt.savefig(out4, dpi=150, bbox_inches='tight')
    plt.close()
    plot_index += 1
    total_plots += 1

    # Plot 5: Month-mean front-intensity map (mean |∇T| over daily fields)
    # Compute average |∇T| across days
    front_stack = np.zeros((nt, ny, nx), dtype=np.float32)
    for k in range(nt):
        front_stack[k] = grad_mag(sst_daily[k])
    front_mean = np.nanmean(front_stack, axis=0)

    plt.figure(figsize=(10, 7))
    im5 = plt.imshow(front_mean, origin='lower', extent=extent, cmap='magma')
    cb5 = plt.colorbar(im5, shrink=0.85, pad=0.02)
    cb5.set_label("Mean |∇T| (°C per grid unit) over month")
    plt.hlines(transect_lat, lon_min, lon_max, colors='white', linestyles='--', linewidth=1.0)
    plt.title(f"{m} Gulf Stream: Month-mean front intensity map (|∇T|)\nQ={quality}, daily stride={stride_hours}h")
    plt.xlabel("Longitude (°E)")
    plt.ylabel("Latitude (°N)")
    out5 = plots_dir / f"plot_{plot_index}_20251202_044455.png"
    plt.savefig(out5, dpi=150, bbox_inches='tight')
    plt.close()
    plot_index += 1
    total_plots += 1

print(f"Created {total_plots} plots successfully")