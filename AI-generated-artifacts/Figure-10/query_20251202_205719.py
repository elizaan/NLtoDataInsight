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
codes_dir = current_file.parent.parent.parent  # Go up to ai_data/codes/../.. → ai_data
project_root = codes_dir.parent.parent  # Go up to agent6-web-app/../.. → NLQtoDataInsight/
sys.path.insert(0, str(project_root))

# Mask module path
mask_module_path = os.path.abspath(os.path.join(project_root, 'LLC2160_land_ocean_mask'))
if mask_module_path not in sys.path:
    sys.path.insert(0, mask_module_path)

try:
    from utils import compute_land_ocean_mask3d
except Exception as e:
    # Fallback: create a dummy ocean mask if mask module is unavailable
    compute_land_ocean_mask3d = None

def squeeze2d(arr):
    a = np.asarray(arr)
    if a.ndim == 3:
        # Common shapes: (z, y, x) or (y, x, z). Prefer to squeeze any singleton dim.
        if a.shape[0] == 1:
            return a[0, ...]
        if a.shape[-1] == 1:
            return a[..., 0]
        # If still 3D, try to reduce by picking the first slice along the smallest dim
        min_axis = int(np.argmin(a.shape))
        indexer = [slice(None)] * a.ndim
        indexer[min_axis] = 0
        return a[tuple(indexer)]
    elif a.ndim == 2:
        return a
    elif a.ndim == 1:
        # Can't plot 1D, make a 2D guess is not possible; raise
        raise ValueError("Expected at least 2D array from dataset read; got 1D")
    else:
        raise ValueError(f"Unsupported array ndim={a.ndim}")

def nanstats(arr):
    return {
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
        "p01": float(np.nanpercentile(arr, 1)),
        "p05": float(np.nanpercentile(arr, 5)),
        "p25": float(np.nanpercentile(arr, 25)),
        "p50": float(np.nanpercentile(arr, 50)),
        "p75": float(np.nanpercentile(arr, 75)),
        "p95": float(np.nanpercentile(arr, 95)),
        "p99": float(np.nanpercentile(arr, 99)),
    }

# Pre-analyzer specifications (use exactly)
quality = 0  # Use this exact value
x_range = [768, 1776]  # Use these exact indices (inclusive)
y_range = [4779, 5310]
z_range = [0, 1]
lat_range = [30.005430221557617, 45.99466323852539]  # For metadata
lon_range = [-5.979166507720947, 35.97916793823242]

# Temporal mapping (dataset uses hourly timesteps)
dataset_start = datetime.strptime("2020-01-20", "%Y-%m-%d")
dataset_end = datetime.strptime("2021-03-26", "%Y-%m-%d")
total_timesteps = 10366  # [0..10365]

# Choose single snapshot likely to show active mesoscale eddies (fall)
chosen_time_utc = datetime(2020, 10, 15, 12, 0)  # 2020-10-15 12:00 UTC
timestep_index = int((chosen_time_utc - dataset_start).total_seconds() // 3600)
timestep_index = max(0, min(timestep_index, total_timesteps - 1))
time_range_meta = [timestep_index, timestep_index, 1]

# Target variables (surface)
target_variables = ["velocity_u_surface", "velocity_v_surface", "temperature_surface"]

# Dataset URLs
u_url = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc2160_arco/visus.idx"
v_url = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc2160_v/v_llc2160_x_y_depth.idx"
temp_url = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc2160_theta/llc2160_theta.idx"

# Prepare ocean mask (surface only)
ocean_mask = None
if compute_land_ocean_mask3d is not None:
    try:
        mask3d = compute_land_ocean_mask3d(
            time=0,  # Land/ocean boundary doesn't change with time
            quality=quality,  # Use same quality as data for alignment
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            use_surface_only=True  # Efficient: land/ocean same at all depths for surface slice
        )
        mask2d = squeeze2d(mask3d)
        ocean_mask = (mask2d == 1)
    except Exception as e:
        ocean_mask = None

# Ensure output directories exist
cache_path = Path("/Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app/ai_data/data_cache/dyamond_llc2160/data_20251202_205719.npz")
cache_path.parent.mkdir(parents=True, exist_ok=True)

# Load datasets
try:
    ds_u = ovp.LoadDataset(u_url)
    ds_v = ovp.LoadDataset(v_url)
    ds_t = ovp.LoadDataset(temp_url)
except Exception as e:
    print(json.dumps({"error": f"Failed to load datasets: {str(e)}"}))
    sys.exit(1)

# Validate timestep exists in datasets (defensive)
try:
    ts_u = ds_u.db.getTimesteps()
    ts_v = ds_v.db.getTimesteps()
    ts_t = ds_t.db.getTimesteps()
    # Clamp to each length just in case
    if len(ts_u) > 0:
        timestep_index = max(0, min(timestep_index, len(ts_u) - 1))
    if len(ts_v) > 0:
        timestep_index = max(0, min(timestep_index, len(ts_v) - 1))
    if len(ts_t) > 0:
        timestep_index = max(0, min(timestep_index, len(ts_t) - 1))
except Exception:
    pass

# Read data for the chosen timestep (surface layer)
try:
    u_raw = ds_u.db.read(
        time=timestep_index,
        x=[x_range[0], x_range[1]],
        y=[y_range[0], y_range[1]],
        z=[z_range[0], z_range[1]],
        quality=quality
    )
    v_raw = ds_v.db.read(
        time=timestep_index,
        x=[x_range[0], x_range[1]],
        y=[y_range[0], y_range[1]],
        z=[z_range[0], z_range[1]],
        quality=quality
    )
    t_raw = ds_t.db.read(
        time=timestep_index,
        x=[x_range[0], x_range[1]],
        y=[y_range[0], y_range[1]],
        z=[z_range[0], z_range[1]],
        quality=quality
    )
except Exception as e:
    print(json.dumps({"error": f"Failed to read data arrays: {str(e)}"}))
    sys.exit(1)

# Convert to 2D arrays (surface)
try:
    u2d = squeeze2d(u_raw).astype(np.float32)
    v2d = squeeze2d(v_raw).astype(np.float32)
    t2d = squeeze2d(t_raw).astype(np.float32)
except Exception as e:
    print(json.dumps({"error": f"Failed to convert arrays to 2D: {str(e)}"}))
    sys.exit(1)

ny, nx = u2d.shape  # Expect (y, x)

# Apply ocean mask
if ocean_mask is None:
    # If mask is unavailable, consider everything ocean for analysis
    ocean_mask2d = np.ones_like(u2d, dtype=bool)
else:
    ocean_mask2d = ocean_mask.astype(bool)
    # Ensure shapes match
    if ocean_mask2d.shape != u2d.shape:
        # Attempt to resize via simple cropping or broadcasting (conservative approach)
        min_ny = min(ocean_mask2d.shape[0], ny)
        min_nx = min(ocean_mask2d.shape[1], nx)
        ocean_mask2d = ocean_mask2d[:min_ny, :min_nx]
        u2d = u2d[:min_ny, :min_nx]
        v2d = v2d[:min_ny, :min_nx]
        t2d = t2d[:min_ny, :min_nx]
        ny, nx = ocean_mask2d.shape

# Mask land with NaNs
u = np.where(ocean_mask2d, u2d, np.nan)
v = np.where(ocean_mask2d, v2d, np.nan)
temp = np.where(ocean_mask2d, t2d, np.nan)

# Derived fields
speed = np.sqrt(u**2 + v**2)

# Build simple lat/lon coordinate vectors for extent and metric approximations
lat_min, lat_max = lat_range
lon_min, lon_max = lon_range
lat_vec = np.linspace(lat_min, lat_max, ny).astype(np.float32)
lon_vec = np.linspace(lon_min, lon_max, nx).astype(np.float32)

# Compute relative vorticity ζ = dv/dx − du/dy (approximate metric on sphere)
# Use average degree spacing converted to meters, with cos(lat) factor for dx
R_earth = 6371000.0  # meters
# Mean spacings in degrees
dlat_deg = (lat_max - lat_min) / max(1, (ny - 1))
dlon_deg = (lon_max - lon_min) / max(1, (nx - 1))

# Convert to meters
dy_m = np.deg2rad(dlat_deg) * R_earth  # constant across the domain
# dx varies with latitude: dx = R * cos(lat) * dlon_rad
dx_per_row = (np.deg2rad(dlon_deg) * R_earth * np.cos(np.deg2rad(lat_vec))).astype(np.float64)  # shape (ny,)

# Gradients using central differences via numpy.gradient (grid indexing: axis 0 -> y/lat, axis 1 -> x/lon)
# numpy.gradient returns derivative w.r.t. index; scale by physical spacing.
# For dx, apply per-row scaling; for dy, divide by dy_m constant.
with np.errstate(invalid='ignore'):
    dv_dxi = np.gradient(v, axis=1)  # shape (ny, nx)
    du_deta = np.gradient(u, axis=0)  # shape (ny, nx)
    # Avoid division by zero by masking rows with near-zero dx
    valid_rows = np.abs(dx_per_row) > 0
    dvdx = np.full_like(v, np.nan, dtype=np.float64)
    dudy = np.full_like(u, np.nan, dtype=np.float64)
    dvdx[valid_rows, :] = dv_dxi[valid_rows, :] / dx_per_row[valid_rows, None]
    if dy_m != 0:
        dudy[:, :] = du_deta[:, :] / dy_m
    vorticity = (dvdx - dudy).astype(np.float32)

# SST anomaly (temperature minus Mediterranean-domain mean at this timestep)
sst_mean = float(np.nanmean(temp))
sst_anom = (temp - sst_mean).astype(np.float32)

# Summary statistics
stats = {
    "speed": nanstats(speed),
    "vorticity": nanstats(vorticity),
    "sst_anomaly": nanstats(sst_anom),
    "u": nanstats(u),
    "v": nanstats(v),
    "temperature": nanstats(temp),
}

# Data points processed (sum across variables, counting only in-domain 2D arrays)
data_points_processed = int(u.size + v.size + temp.size)

# Save for plotting
np.savez(
    cache_path,
    u=u.astype(np.float32),
    v=v.astype(np.float32),
    temp=temp.astype(np.float32),
    speed=speed.astype(np.float32),
    vorticity=vorticity.astype(np.float32),
    sst_anom=sst_anom.astype(np.float32),
    ocean_mask=ocean_mask2d.astype(np.uint8),
    x_range=np.array(x_range, dtype=np.int32),
    y_range=np.array(y_range, dtype=np.int32),
    z_range=np.array(z_range, dtype=np.int32),
    time_range=np.array(time_range_meta, dtype=np.int32),
    lat_range=np.array(lat_range, dtype=np.float64),
    lon_range=np.array(lon_range, dtype=np.float64),
    lat_vec=lat_vec,
    lon_vec=lon_vec,
    chosen_time_utc=np.array([chosen_time_utc.strftime("%Y-%m-%d %H:%M:%S")]),
    timestep_index=np.array([timestep_index], dtype=np.int32),
    quality_level=np.array([quality], dtype=np.int32),
    target_variables=np.array(target_variables),
    dataset_start=np.array([dataset_start.strftime("%Y-%m-%d %H:%M:%S")]),
    dataset_end=np.array([dataset_end.strftime("%Y-%m-%d %H:%M:%S")]),
)

# Output JSON summary
print(json.dumps({
    "status": "success",
    "message": "Single-snapshot surface extraction complete for Mediterranean Sea.",
    "chosen_time_utc": chosen_time_utc.strftime("%Y-%m-%d %H:%M:%S"),
    "timestep_index": timestep_index,
    "x_range": x_range,
    "y_range": y_range,
    "z_range": z_range,
    "lat_range": lat_range,
    "lon_range": lon_range,
    "quality_level": quality,
    "variables": target_variables + ["speed", "vorticity", "sst_anomaly"],
    "data_points_processed": data_points_processed,
    "stats": stats,
    "cache_file": str(cache_path)
}))