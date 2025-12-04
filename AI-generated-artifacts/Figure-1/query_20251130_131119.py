import openvisuspy as ovp
import numpy as np
import json
from datetime import datetime, timedelta
import sys, os
from pathlib import Path

# -------------------------------
# Project paths (as provided)
# -------------------------------
current_file = Path(__file__)  # query_*.py
codes_dir = current_file.parent.parent.parent  # ai_data
project_root = codes_dir.parent.parent  # NLQtoDataInsight
sys.path.insert(0, str(project_root))

# Land/ocean mask module
mask_module_path = os.path.abspath(os.path.join(project_root, 'LLC2160_land_ocean_mask'))
if mask_module_path not in sys.path:
    sys.path.insert(0, mask_module_path)

from utils import compute_land_ocean_mask3d

def main():
    # -------------------------------
    # Pre-analyzer fixed specifications (USE EXACTLY)
    # -------------------------------
    quality = 0  # Full resolution
    x_range = [0, 8640]
    y_range = [0, 6480]
    z_range = [0, 1]  # Using surface slice as per guidance
    lat_range = [-89.9947280883789, 72.03472137451172]
    lon_range = [-180.0, 179.99996948242188]
    target_variable = "Temperature"
    dataset_url = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc2160_theta/llc2160_theta.idx"

    # Dataset temporal info
    dataset_start_str = "2020-01-20"
    dataset_end_str = "2021-03-26"
    total_timesteps = 10366  # inclusive count
    time_unit = "hours"

    # -------------------------------
    # User request temporal mapping
    # -------------------------------
    # User asked: "global temperature on January 20, 2020"
    # We interpret as the first available hour on that date: 2020-01-20 00:00
    dataset_start = datetime.strptime(dataset_start_str, "%Y-%m-%d")
    dataset_end = datetime.strptime(dataset_end_str, "%Y-%m-%d")

    user_date_str = "2020-01-20"
    user_start = datetime.strptime(user_date_str, "%Y-%m-%d")  # 00:00 of the day
    user_end = user_start  # single instant for the map

    # Calculate timestep indices (hourly resolution)
    timestep_start = int((user_start - dataset_start).total_seconds() / 3600)
    timestep_end = int((user_end - dataset_start).total_seconds() / 3600)

    # Clamp to dataset range
    timestep_start = max(0, min(timestep_start, total_timesteps - 1))
    timestep_end = max(0, min(timestep_end, total_timesteps - 1))
    timestep_step = 1

    # For this request, we only need the first timestep on 2020-01-20 (t=0)
    timesteps = list(range(timestep_start, timestep_end + 1, timestep_step))

    # -------------------------------
    # Prepare mask (surface-only)
    # -------------------------------
    mask3d = compute_land_ocean_mask3d(
        time=0,  # Land/ocean does not change with time
        quality=quality,
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        use_surface_only=True
    )

    if mask3d.ndim == 3:
        ocean_mask = (mask3d[0] == 1)
    elif mask3d.ndim == 2:
        ocean_mask = (mask3d == 1)
    else:
        raise RuntimeError("Unexpected mask dimensionality")

    # -------------------------------
    # Load dataset
    # -------------------------------
    ds = ovp.LoadDataset(dataset_url)
    length_of_timesteps = len(ds.db.getTimesteps())

    # -------------------------------
    # Read data for the requested timestep(s)
    # -------------------------------
    results = []
    saved_arrays = {}
    data_points_processed = 0

    for t in timesteps:
        # Read full global, surface slice at Q=0
        data = ds.db.read(
            time=t,
            x=[x_range[0], x_range[1]],
            y=[y_range[0], y_range[1]],
            z=[z_range[0], z_range[1]],
            quality=quality
        )

        arr = np.asarray(data)
        # Reduce to 2D surface (y,x) if needed
        if arr.ndim == 3:
            arr = arr[0, :, :]  # use surface layer z=0
        elif arr.ndim == 2:
            pass
        else:
            raise RuntimeError(f"Unexpected array shape {arr.shape}")

        # Apply ocean mask (land -> NaN)
        # Ensure mask shape matches (y,x)
        if ocean_mask.shape != arr.shape:
            raise RuntimeError(f"Mask shape {ocean_mask.shape} does not match data shape {arr.shape}")

        arr = np.where(ocean_mask, arr, np.nan).astype(np.float32, copy=False)

        data_points_processed = int(arr.size)

        # Stats (ocean-only, NaN-safe)
        stats = {
            "timestep": int(t),
            "datetime": (dataset_start + timedelta(hours=int(t))).strftime("%Y-%m-%d %H:%M:%S"),
            "mean": float(np.nanmean(arr)),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "std": float(np.nanstd(arr)),
            "unit": "degrees Celsius"
        }
        results.append(stats)

        # Save the actual 2D field for plotting
        saved_arrays[f"temperature_surface_t{t}"] = arr

    # -------------------------------
    # Save to cache (.npz)
    # -------------------------------
    cache_path = Path("/Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app/ai_data/data_cache/dyamond_llc2160/data_20251130_131119.npz")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Build metadata to persist along with arrays
    meta = {
        "variable_names": ["Temperature"],
        "x_range": np.array(x_range, dtype=np.int64),
        "y_range": np.array(y_range, dtype=np.int64),
        "z_range": np.array(z_range, dtype=np.int64),
        "time_range": np.array([timestep_start, timestep_end, timestep_step], dtype=np.int64),
        "lat_range": np.array(lat_range, dtype=np.float64),
        "lon_range": np.array(lon_range, dtype=np.float64),
        "quality_level": np.array([quality], dtype=np.int64),
        "time_unit": np.array([time_unit]),
        "dataset_start": np.array([dataset_start.strftime("%Y-%m-%d %H:%M:%S")]),
        "dataset_end": np.array([dataset_end.strftime("%Y-%m-%d %H:%M:%S")]),
        "requested_date": np.array([user_date_str]),
        "results_json": np.array([json.dumps(results)])
    }

    # Use compressed to keep file size manageable
    np.savez_compressed(
        cache_path,
        **saved_arrays,
        **meta
    )

    # -------------------------------
    # Print JSON summary to stdout
    # -------------------------------
    print(json.dumps({
        "status": "success",
        "variable_names": ["Temperature"],
        "strategy": "Full-resolution (Q=0) global surface slice at hourly timestep corresponding to 2020-01-20 00:00; ocean mask applied; statistics computed and full 2D field cached for plotting.",
        "data_points_processed": data_points_processed,
        "x_range": x_range,
        "y_range": y_range,
        "z_range": z_range,
        "time_range": [timestep_start, timestep_end, timestep_step],
        "lat_range": lat_range,
        "lon_range": lon_range,
        "quality_level": quality,
        "timesteps_len_in_dataset": length_of_timesteps,
        "results": results,
        "cache_file": str(cache_path)
    }))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(json.dumps({"error": str(e)}))