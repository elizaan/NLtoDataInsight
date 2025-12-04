import numpy as np
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path

def main():
    # Paths
    cache_file = "/Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app/ai_data/data_cache/dyamond_llc2160/data_20251130_131119.npz"
    plots_dir = "/Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app/ai_data/plots/dyamond_llc2160"
    os.makedirs(plots_dir, exist_ok=True)

    # Load cached data
    data = np.load(cache_file, allow_pickle=True)
    print("Available keys:", data.files)

    # Extract metadata safely
    x_range = data["x_range"].tolist() if "x_range" in data else [0, 8640]
    y_range = data["y_range"].tolist() if "y_range" in data else [0, 6480]
    lat_range = data["lat_range"].tolist() if "lat_range" in data else None
    lon_range = data["lon_range"].tolist() if "lon_range" in data else None
    quality = int(data["quality_level"][0]) if "quality_level" in data else 0
    dataset_start_str = str(data["dataset_start"][0]) if "dataset_start" in data else "2020-01-20 00:00:00"
    requested_date = str(data["requested_date"][0]) if "requested_date" in data else "2020-01-20"
    time_range = data["time_range"].tolist() if "time_range" in data else [0, 0, 1]
    results = None
    if "results_json" in data:
        try:
            results = json.loads(data["results_json"][0])
        except Exception:
            results = None

    # Identify the temperature array key saved by query code
    temp_keys = [k for k in data.files if k.startswith("temperature_surface_t")]
    if not temp_keys:
        raise RuntimeError("No temperature surface arrays found in the cache. Keys: {}".format(data.files))
    temp_key = sorted(temp_keys)[0]  # e.g., 'temperature_surface_t0'
    temp = data[temp_key]

    # Compute stats (nan-aware)
    valid = temp[~np.isnan(temp)]
    vmin = float(np.nanmin(temp))
    vmax = float(np.nanmax(temp))
    vmean = float(np.nanmean(temp))
    vstd = float(np.nanstd(temp))

    # Get human-readable timestamp if available
    ts_label = None
    if results and isinstance(results, list) and len(results) > 0 and "datetime" in results[0]:
        ts_label = results[0]["datetime"]
    else:
        # Fallback: derive from time_range and dataset_start
        ts_label = f"{requested_date} 00:00:00"

    # Plot 1: Global temperature map (LLC2160 model grid indices)
    # Using origin='lower' so row 0 is at the bottom, as required
    plt.figure(figsize=(14, 7))
    im = plt.imshow(
        temp,
        origin='lower',  # row-0 at the bottom
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
        cmap='turbo',
        vmin=vmin,
        vmax=vmax,
        aspect='auto'
    )
    cbar = plt.colorbar(im, pad=0.01)
    cbar.set_label("Temperature (°C)")
    plt.title(f"DYAMOND LLC2160 Surface Temperature (z=0)\n{ts_label} UTC | Q={quality} | Grid indices (not regular lat/lon)")
    plt.xlabel("Model grid X index")
    plt.ylabel("Model grid Y index")
    # Annotate key stats
    plt.text(0.01, 0.01,
             f"Mean: {vmean:.2f} °C  |  Min: {vmin:.2f} °C  |  Max: {vmax:.2f} °C  |  Std: {vstd:.2f} °C",
             transform=plt.gca().transAxes, color='white',
             bbox=dict(facecolor='black', alpha=0.4, pad=4, edgecolor='none'))
    out1 = os.path.join(plots_dir, "plot_1_20251130_131119.png")
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Histogram of global ocean surface temperatures
    plt.figure(figsize=(10, 6))
    # Use a reasonable number of bins given large dataset
    bins = 200
    plt.hist(valid, bins=bins, color='steelblue', edgecolor='none', alpha=0.9)
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Number of grid cells")
    plt.title(f"Global Ocean Surface Temperature Distribution\n{ts_label} UTC")
    # Set x-limits to observed min/max
    plt.xlim(vmin, vmax)
    # Add vertical lines for mean and ±1 std
    plt.axvline(vmean, color='red', linestyle='--', linewidth=1.5, label=f"Mean = {vmean:.2f} °C")
    plt.axvline(vmean - vstd, color='orange', linestyle=':', linewidth=1.2, label=f"Mean ± 1σ ({vstd:.2f} °C)")
    plt.axvline(vmean + vstd, color='orange', linestyle=':', linewidth=1.2)
    plt.legend()
    out2 = os.path.join(plots_dir, "plot_2_20251130_131119.png")
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close()

    print("Created 2 plots successfully")

if __name__ == "__main__":
    main()