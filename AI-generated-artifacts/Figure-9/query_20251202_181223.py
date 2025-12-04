import openvisuspy as ovp
import numpy as np
import json
from datetime import datetime, timedelta
import sys, os
from pathlib import Path

# Add project root to Python path
# From: agent6-web-app/ai_data/codes/dyamond_llc2160/query_*.py
# To:   NLQtoDataInsight/ (project root)
current_file = Path(__file__)  # query_*.py
codes_dir = current_file.parent.parent.parent  # Go up to ai_data
project_root = codes_dir.parent.parent  # Go up to NLQtoDataInsight/
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Land/ocean mask helper
mask_module_path = os.path.abspath(os.path.join(project_root, 'LLC2160_land_ocean_mask'))
if mask_module_path not in sys.path:
    sys.path.insert(0, mask_module_path)
from utils import compute_land_ocean_mask3d

def main():
    # Fixed parameters from pre-analyzer specs
    quality = 0  # Use this exact value
    x_range = [1680, 1968]  # Use these exact indices
    y_range = [4274, 4779]
    z_range = [0, 1]  # surface layer only
    lat_range = [12.034605979919434, 29.97222900390625]  # For metadata
    lon_range = [32.02083206176758, 43.97916793823242]
    salinity_url = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc2160_salt/salt_llc2160_x_y_depth.idx"
    variable_name = "Salinity"
    units = "g kg-1"

    # Temporal mapping: Single hourly snapshot at 2020-02-01 00:00 UTC
    dataset_start = datetime.strptime("2020-01-20", "%Y-%m-%d")
    requested_time = datetime.strptime("2020-02-01 00:00", "%Y-%m-%d %H:%M")
    timestep_index = int((requested_time - dataset_start).total_seconds() / 3600.0)
    # Clamp to valid range if needed (dataset total 10366)
    timestep_index = max(0, min(timestep_index, 10366 - 1))
    YOUR_START_TIME = timestep_index
    YOUR_END_TIME = timestep_index
    YOUR_STEP = 1
    timestep_range = [timestep_index]

    # Prepare ocean mask (surface-only is sufficient and constant in time)
    mask3d = compute_land_ocean_mask3d(
        time=0,  # Land/ocean boundary doesn't change with time
        quality=quality,  # Use same quality as data for alignment
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        use_surface_only=True  # Efficient: land/ocean same at all depths
    )
    # Ensure 2D mask for the surface
    if mask3d.ndim == 3:
        ocean_mask_2d = (mask3d[0] == 1)
    else:
        ocean_mask_2d = (mask3d == 1)

    try:
        # Load dataset
        ds = ovp.LoadDataset(salinity_url)
        # Validate timesteps exist
        _timesteps = ds.db.getTimesteps()
        length_of_timesteps = len(_timesteps) if _timesteps is not None else None

        data_points_processed = 0
        results = []
        saved_arrays = {}

        for t in timestep_range:
            raw = ds.db.read(
                time=t,
                x=[x_range[0], x_range[1]],
                y=[y_range[0], y_range[1]],
                z=[z_range[0], z_range[1]],
                quality=quality
            )

            arr = np.array(raw)
            # Squeeze any singleton dimension; expect 2D [y, x] or [x, y]
            arr = np.squeeze(arr)

            # If array comes as [x, y], transpose to [y, x] for consistency
            if arr.ndim == 2 and arr.shape[0] == (x_range[1] - x_range[0] + 1) and arr.shape[1] == (y_range[1] - y_range[0] + 1):
                arr = arr.T  # now [y, x]

            if arr.ndim != 2:
                # Fallback: try to bring to 2D [y, x]
                # Common cases: (1, Y, X) or (Y, X, 1)
                if arr.ndim == 3:
                    # pick first along any singleton dim
                    arr = np.squeeze(arr)
                if arr.ndim != 2:
                    raise RuntimeError(f"Unexpected array shape after squeeze: {arr.shape}")

            # Ensure mask shape matches data shape
            if ocean_mask_2d.shape != arr.shape:
                # Attempt to reshape or transpose if needed
                if ocean_mask_2d.T.shape == arr.shape:
                    ocean_mask = ocean_mask_2d.T
                else:
                    # Last resort: broadcast if possible
                    try:
                        ocean_mask = np.broadcast_to(ocean_mask_2d, arr.shape)
                    except Exception:
                        raise RuntimeError(f"Mask shape {ocean_mask_2d.shape} not compatible with data shape {arr.shape}")
            else:
                ocean_mask = ocean_mask_2d

            # Mask out land as NaN
            sal2d = np.where(ocean_mask, arr, np.nan)

            # Stats on ocean-only
            vmin = np.nanmin(sal2d)
            vmax = np.nanmax(sal2d)
            vmean = np.nanmean(sal2d)
            vstd = np.nanstd(sal2d)

            # Histogram (40 bins) over finite values
            finite_vals = sal2d[np.isfinite(sal2d)]
            if finite_vals.size > 0:
                hist_counts, hist_edges = np.histogram(finite_vals, bins=40)
                # Percentiles
                percentile_labels = ["p10", "p25", "p50", "p75", "p90", "p95", "p99"]
                percentile_values = np.percentile(finite_vals, [10, 25, 50, 75, 90, 95, 99])
            else:
                hist_counts = np.array([])
                hist_edges = np.array([])
                percentile_labels = ["p10", "p25", "p50", "p75", "p90", "p95", "p99"]
                percentile_values = np.array([np.nan] * len(percentile_labels))

            data_points_processed = sal2d.size  # grid points in subdomain (including land)
            ocean_points = int(np.isfinite(sal2d).sum())

            your_finding = {
                "timestep": int(t),
                "time_iso": (dataset_start + timedelta(hours=int(t))).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "mean": float(vmean),
                "min": float(vmin),
                "max": float(vmax),
                "std": float(vstd),
                "ocean_points": ocean_points,
                "total_grid_points": int(sal2d.size),
                "histogram_bins": 40,
                "percentiles": {k: float(v) for k, v in zip(percentile_labels, percentile_values.tolist())}
            }
            results.append(your_finding)

            # Save arrays for plotting phase
            saved_arrays["salinity_surface"] = sal2d.astype(np.float32)
            saved_arrays["ocean_mask"] = ocean_mask.astype(np.uint8)
            saved_arrays["hist_counts"] = hist_counts.astype(np.int64)
            saved_arrays["hist_edges"] = hist_edges.astype(np.float32)
            saved_arrays["percentiles"] = percentile_values.astype(np.float32)
            saved_arrays["percentiles_labels"] = np.array(percentile_labels)

        # Prepare save path
        save_path = '/Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app/ai_data/data_cache/dyamond_llc2160/data_20251202_181223.npz'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save for plotting with rich metadata
        np.savez(
            save_path,
            variable_names=np.array([variable_name]),
            salinity_units=np.array([units]),
            x_range=np.array(x_range, dtype=np.int32),
            y_range=np.array(y_range, dtype=np.int32),
            z_range=np.array(z_range, dtype=np.int32),
            time_range=np.array([YOUR_START_TIME, YOUR_END_TIME, YOUR_STEP], dtype=np.int32),
            time_index=np.array([timestep_index], dtype=np.int32),
            time_iso=np.array([(dataset_start + timedelta(hours=int(timestep_index))).strftime("%Y-%m-%d %H:%M:%S UTC")]),
            lat_range=np.array(lat_range, dtype=np.float64),
            lon_range=np.array(lon_range, dtype=np.float64),
            quality_level=np.array([quality], dtype=np.int32),
            stat_min=np.array([results[0]["min"]], dtype=np.float32),
            stat_max=np.array([results[0]["max"]], dtype=np.float32),
            stat_mean=np.array([results[0]["mean"]], dtype=np.float32),
            stat_std=np.array([results[0]["std"]], dtype=np.float32),
            **saved_arrays
        )

        # Output JSON summary
        print(json.dumps({
            "status": "success",
            "variable_name(s)": variable_name,
            "strategy": "Single hourly snapshot at 2020-02-01 00:00 UTC (timestep index 288), full-resolution (Q=0) surface layer over Red Sea subdomain; apply ocean mask; compute stats, histogram (40 bins), and percentiles; save data and metadata for plotting.",
            "dataset_url": salinity_url,
            "data_points_processed": int(data_points_processed),
            "x_range": x_range,
            "y_range": y_range,
            "z_range": z_range,
            "time_range": [YOUR_START_TIME, YOUR_END_TIME, YOUR_STEP],
            "time_index": timestep_index,
            "time_iso": (dataset_start + timedelta(hours=int(timestep_index))).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "lat_range": lat_range,
            "lon_range": lon_range,
            "quality_level": quality,
            "results": results
        }))

    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()