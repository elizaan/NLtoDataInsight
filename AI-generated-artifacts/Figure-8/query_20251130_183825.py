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
codes_dir = current_file.parent.parent.parent  # .../ai_data
project_root = codes_dir.parent.parent  # .../NLQtoDataInsight
sys.path.insert(0, str(project_root))
# mask module path 
mask_module_path = os.path.abspath(os.path.join(project_root, 'LLC2160_land_ocean_mask'))
if mask_module_path not in sys.path:
    sys.path.insert(0, mask_module_path)
from utils import compute_land_ocean_mask3d

# =========================
# User/task-aligned settings
# =========================
# Use EXACT values from PRE-ANALYZER SPECIFICATIONS
quality = 0
x_range = [1272, 1632]
y_range = [2822, 3141]
z_range = [0, 1]  # We'll read as specified, but use surface slice (z=0) for 2D plots
lat_range = [-39.99394989013672, -30.005430221557617]
lon_range = [15.020833015441895, 29.97916603088379]

# Dataset temporal info
dataset_start = datetime.strptime("2020-01-20", "%Y-%m-%d")
dataset_end = datetime.strptime("2021-03-26", "%Y-%m-%d")
total_timesteps_metadata = 10366  # Provided meta; we will also query ds to confirm

# Target variables and their sources (OpenVisus IDX)
url_temp = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc2160_theta/llc2160_theta.idx"
url_u    = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc2160_arco/visus.idx"
url_v    = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc2160_v/v_llc2160_x_y_depth.idx"

# Output cache path
cache_path = '/Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app/ai_data/data_cache/dyamond_llc2160/data_20251130_183825.npz'

# Helper: add months without external deps
def days_in_month(year, month):
    if month in (1,3,5,7,8,10,12):
        return 31
    if month in (4,6,9,11):
        return 30
    # February
    is_leap = (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
    return 29 if is_leap else 28

def add_months(dt, months):
    y = dt.year + (dt.month - 1 + months) // 12
    m = (dt.month - 1 + months) % 12 + 1
    d = min(dt.day, days_in_month(y, m))
    return dt.replace(year=y, month=m, day=d)

# Build 14 monthly timestamps: one timestep per month starting 2020-01-20 through 2021-02-20 (inclusive)
requested_datetimes = []
for i in range(14):
    dt = add_months(dataset_start, i)  # keep same day-of-month (20th) at 00:00
    # Ensure within dataset range (clamp, though all 14 are within range)
    if dt < dataset_start:
        dt = dataset_start
    if dt > dataset_end:
        dt = dataset_end
    requested_datetimes.append(dt)

# Convert to timestep indices (hours since dataset_start), clamp to [0, total_timesteps-1]
def dt_to_timestep(dt, total_timesteps):
    t = int((dt - dataset_start).total_seconds() / 3600.0)
    return max(0, min(t, total_timesteps - 1))

try:
    # Load datasets
    ds_temp = ovp.LoadDataset(url_temp)
    ds_u = ovp.LoadDataset(url_u)
    ds_v = ovp.LoadDataset(url_v)

    # Determine available timesteps from one dataset (assume synchronized across fields)
    timesteps_list = ds_temp.db.getTimesteps()
    total_timesteps = len(timesteps_list) if timesteps_list is not None else total_timesteps_metadata
    # Map datetimes to integer timesteps
    selected_timesteps = []
    for dt in requested_datetimes:
        selected_timesteps.append(dt_to_timestep(dt, total_timesteps))

    # Compute land/ocean mask once (surface only flag for speed; still matching our ROI)
    mask3d = compute_land_ocean_mask3d(
        time=0,  # Land/sea mask is static
        quality=quality,
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        use_surface_only=True
    )
    # Ensure boolean mask where True indicates ocean
    ocean_mask = (mask3d == 1)
    # Use surface slice for 2D plotting
    if ocean_mask.ndim == 3:
        ocean_mask_2d = ocean_mask[0, :, :]
    else:
        # If already 2D
        ocean_mask_2d = ocean_mask

    # Prepare accumulators
    temps_2d = []
    u_2d = []
    v_2d = []
    dt_strings = []
    results = []

    # Loop over selected monthly timesteps
    for dt_obj, t in zip(requested_datetimes, selected_timesteps):

        # Read temperature
        temp_arr = ds_temp.db.read(
            time=t,
            x=[x_range[0], x_range[1]],
            y=[y_range[0], y_range[1]],
            z=[z_range[0], z_range[1]],
            quality=quality
        )
        # Read u
        u_arr = ds_u.db.read(
            time=t,
            x=[x_range[0], x_range[1]],
            y=[y_range[0], y_range[1]],
            z=[z_range[0], z_range[1]],
            quality=quality
        )
        # Read v
        v_arr = ds_v.db.read(
            time=t,
            x=[x_range[0], x_range[1]],
            y=[y_range[0], y_range[1]],
            z=[z_range[0], z_range[1]],
            quality=quality
        )

        # Slice to surface 2D (z=0), aligning dimensions as [z,y,x]
        if temp_arr.ndim == 3:
            temp2d = temp_arr[0, :, :]
        elif temp_arr.ndim == 2:
            temp2d = temp_arr
        else:
            raise RuntimeError("Unexpected temperature array shape")

        if u_arr.ndim == 3:
            u2d = u_arr[0, :, :]
        elif u_arr.ndim == 2:
            u2d = u_arr
        else:
            raise RuntimeError("Unexpected U array shape")

        if v_arr.ndim == 3:
            v2d = v_arr[0, :, :]
        elif v_arr.ndim == 2:
            v2d = v_arr
        else:
            raise RuntimeError("Unexpected V array shape")

        # Apply ocean mask (NaN on land)
        # Ensure mask shape matches 2D fields
        if ocean_mask_2d.shape != temp2d.shape:
            raise RuntimeError(f"Mask shape {ocean_mask_2d.shape} does not match data shape {temp2d.shape}")
        temp2d = np.where(ocean_mask_2d, temp2d, np.nan)
        u2d = np.where(ocean_mask_2d, u2d, np.nan)
        v2d = np.where(ocean_mask_2d, v2d, np.nan)

        # Compute speed and statistics (surface)
        speed2d = np.sqrt(u2d**2 + v2d**2)
        # Stats excluding NaNs
        temp_mean = float(np.nanmean(temp2d))
        temp_min = float(np.nanmin(temp2d))
        temp_max = float(np.nanmax(temp2d))
        temp_std = float(np.nanstd(temp2d))
        spd_mean = float(np.nanmean(speed2d))
        spd_min = float(np.nanmin(speed2d))
        spd_max = float(np.nanmax(speed2d))
        spd_std = float(np.nanstd(speed2d))

        # Accumulate arrays and metadata
        temps_2d.append(temp2d.astype(np.float32))
        u_2d.append(u2d.astype(np.float32))
        v_2d.append(v2d.astype(np.float32))
        dt_strings.append(dt_obj.strftime("%Y-%m-%d %H:%M:%S"))

        results.append({
            "timestep": int(t),
            "datetime": dt_strings[-1],
            "temperature_mean": temp_mean,
            "temperature_min": temp_min,
            "temperature_max": temp_max,
            "temperature_std": temp_std,
            "speed_mean": spd_mean,
            "speed_min": spd_min,
            "speed_max": spd_max,
            "speed_std": spd_std
        })

    # Stack into (ntime, ny, nx)
    temps_2d = np.stack(temps_2d, axis=0)
    u_2d = np.stack(u_2d, axis=0)
    v_2d = np.stack(v_2d, axis=0)
    timesteps_array = np.array(selected_timesteps, dtype=np.int32)
    dt_strings_array = np.array(dt_strings)

    # Count data points processed (approx total for the three variables)
    data_points_processed = int(temps_2d.size + u_2d.size + v_2d.size)

    # Save to NPZ for plotting
    save_dict = {
        "variable_names": np.array(["Temperature(theta)", "Velocity(u,v)"]),
        "temperature_surface": temps_2d,
        "u_surface": u_2d,
        "v_surface": v_2d,
        "ocean_mask_surface": ocean_mask_2d.astype(np.uint8),
        "timesteps": timesteps_array,
        "datetimes": dt_strings_array,
        "x_range": np.array(x_range, dtype=np.int32),
        "y_range": np.array(y_range, dtype=np.int32),
        "z_range": np.array(z_range, dtype=np.int32),
        "lat_range": np.array(lat_range, dtype=np.float64),
        "lon_range": np.array(lon_range, dtype=np.float64),
        "quality_level": np.array([quality], dtype=np.int32),
        "strategy": np.array(["Monthly sampling: one timestep per month (surface z=0)"], dtype=object),
        "data_points_processed": np.array([data_points_processed], dtype=np.int64),
        "results_json": np.array([json.dumps(results)], dtype=object)
    }
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, **save_dict)

    # Output JSON summary
    print(json.dumps({
        "status": "success",
        "variable_names": ["Temperature(theta)", "Velocity(u,v)"],
        "strategy": "Monthly sampling (14 months) at z=0 surface within Agulhas ROI; apply ocean mask; store temperature and velocity (u,v) for plotting heatmaps and vector fields.",
        "data_points_processed": data_points_processed,
        "timesteps": selected_timesteps,
        "datetimes": dt_strings,
        "x_range": x_range,
        "y_range": y_range,
        "z_range": z_range,
        "lat_range": lat_range,
        "lon_range": lon_range,
        "quality_level": quality,
        "results": results,
        "cache_path": cache_path
    }))

except Exception as e:
    print(json.dumps({"status": "error", "error": str(e)}))