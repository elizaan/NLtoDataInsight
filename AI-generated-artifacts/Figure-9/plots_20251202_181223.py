import os
import numpy as np
import matplotlib.pyplot as plt
import json

# Optional mapping overlay
try:
    import cartopy.crs as ccrs
    CARTOPY_AVAILABLE = True
except Exception:
    CARTOPY_AVAILABLE = False

def main():
    cache_path = '/Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app/ai_data/data_cache/dyamond_llc2160/data_20251202_181223.npz'
    out_dir = '/Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app/ai_data/plots/dyamond_llc2160'
    os.makedirs(out_dir, exist_ok=True)

    data = np.load(cache_path, allow_pickle=True)
    print("Available keys in cached NPZ:", data.files)

    # Extract metadata and arrays
    sal = data['salinity_surface']  # 2D array [y, x], ocean as finite, land as NaN
    ocean_mask = data['ocean_mask']
    x_range = data['x_range'].tolist()
    y_range = data['y_range'].tolist()
    z_range = data['z_range'].tolist()
    time_iso = str(data['time_iso'][0])
    quality = int(data['quality_level'][0])
    lat_range = data['lat_range'].tolist()
    lon_range = data['lon_range'].tolist()
    var_names = data['variable_names'].tolist()
    units = str(data['salinity_units'][0]) if 'salinity_units' in data else 'g kg-1'

    # Stats (precomputed)
    vmin = float(data['stat_min'][0]) if 'stat_min' in data else float(np.nanmin(sal))
    vmax = float(data['stat_max'][0]) if 'stat_max' in data else float(np.nanmax(sal))
    vmean = float(data['stat_mean'][0]) if 'stat_mean' in data else float(np.nanmean(sal))
    vstd = float(data['stat_std'][0]) if 'stat_std' in data else float(np.nanstd(sal))

    # Histogram and percentiles (precomputed; fallback to compute if missing)
    if 'hist_counts' in data and 'hist_edges' in data:
        hist_counts = data['hist_counts']
        hist_edges = data['hist_edges']
    else:
        finite_vals = sal[np.isfinite(sal)]
        hist_counts, hist_edges = np.histogram(finite_vals, bins=40)

    if 'percentiles' in data and 'percentiles_labels' in data:
        percentiles = data['percentiles']
        percentile_labels = [str(s) for s in data['percentiles_labels']]
    else:
        finite_vals = sal[np.isfinite(sal)]
        perc_vals = np.percentile(finite_vals, [10, 25, 50, 75, 90, 95, 99]) if finite_vals.size > 0 else np.array([np.nan]*7)
        percentiles = perc_vals.astype(np.float32)
        percentile_labels = ["p10", "p25", "p50", "p75", "p90", "p95", "p99"]

    # Common title parts
    title_time = f"{time_iso}"
    title_suffix = f"(Q={quality}, z=0)"

    # Plot 1: 2D map with coastline overlay (if cartopy available)
    plt.figure(figsize=(10, 12))
    if CARTOPY_AVAILABLE:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], crs=ccrs.PlateCarree())
        im = ax.imshow(
            sal, origin='lower',
            extent=[lon_range[0], lon_range[1], lat_range[0], lat_range[1]],
            transform=ccrs.PlateCarree(),
            cmap='viridis'
        )
        ax.coastlines(resolution='110m', linewidth=0.8)
        ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5, linestyle='--')
        ax.set_title(f"Surface Salinity over Red Sea {title_suffix}\n{title_time}")
    else:
        # Fallback to simple imshow without coastlines, using lon/lat extents
        im = plt.imshow(
            sal, origin='lower',
            extent=[lon_range[0], lon_range[1], lat_range[0], lat_range[1]],
            cmap='viridis', aspect='auto'
        )
        plt.xlabel("Longitude (°E)")
        plt.ylabel("Latitude (°N)")
        plt.title(f"Surface Salinity over Red Sea {title_suffix}\n{title_time}\n(Note: cartopy unavailable, no coastline overlay)")

    cbar = plt.colorbar(im)
    cbar.set_label(f"Salinity ({units})")

    # Annotate key stats
    text_stats = f"Min: {vmin:.3f} {units}\nMean: {vmean:.3f} {units}\nMax: {vmax:.3f} {units}\nStd: {vstd:.3f}"
    # Place in upper-left
    ax_for_text = plt.gca()
    ax_for_text.text(
        0.02, 0.98, text_stats, transform=ax_for_text.transAxes,
        va='top', ha='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'),
        fontsize=10
    )

    out1 = os.path.join(out_dir, 'plot_1_20251202_181223.png')
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Histogram of salinity values (ocean-only)
    bin_centers = 0.5 * (hist_edges[:-1] + hist_edges[1:]) if hist_edges.size > 1 else np.array([])
    plt.figure(figsize=(10, 6))
    if bin_centers.size > 0:
        width = np.diff(hist_edges)
        plt.bar(bin_centers, hist_counts, width=width, align='center', color='#3182bd', edgecolor='white')
        plt.xlabel(f"Salinity ({units})")
        plt.ylabel("Count")
        plt.title(f"Distribution of Surface Salinity over Red Sea {title_suffix}\n{title_time}")
        # Mark mean
        plt.axvline(vmean, color='red', linestyle='--', linewidth=1.5, label=f"Mean = {vmean:.3f}")
        plt.axvline(vmin, color='gray', linestyle=':', linewidth=1.2, label=f"Min = {vmin:.3f}")
        plt.axvline(vmax, color='gray', linestyle=':', linewidth=1.2, label=f"Max = {vmax:.3f}")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No finite ocean data for histogram", ha='center', va='center', transform=plt.gca().transAxes)
        plt.title("Histogram unavailable")

    out2 = os.path.join(out_dir, 'plot_2_20251202_181223.png')
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Percentile summary
    # Create a horizontal bar plot for percentiles
    plt.figure(figsize=(8, 5))
    labels = percentile_labels
    values = percentiles.astype(float)
    y_pos = np.arange(len(labels))
    plt.barh(y_pos, values, color='#2ca25f')
    plt.yticks(y_pos, labels)
    plt.xlabel(f"Salinity ({units})")
    plt.title(f"Surface Salinity Percentiles over Red Sea {title_suffix}\n{title_time}")
    # Annotate values
    for i, v in enumerate(values):
        plt.text(v, i, f" {v:.3f}", va='center', ha='left')

    out3 = os.path.join(out_dir, 'plot_3_20251202_181223.png')
    plt.savefig(out3, dpi=150, bbox_inches='tight')
    plt.close()

    print(json.dumps({
        "status": "success",
        "plots_created": [out1, out2, out3],
        "stats": {
            "min": vmin,
            "mean": vmean,
            "max": vmax,
            "std": vstd
        },
        "time_iso": time_iso,
        "quality_level": quality,
        "lat_range": lat_range,
        "lon_range": lon_range
    }))

if __name__ == "__main__":
    main()