import openvisuspy as ovp
import numpy as np
import json
from datetime import datetime, timedelta
import sys, os
from pathlib import Path

# Add project root to Python path
# From: agent6-web-app/ai_data/codes/dyamond_llc2160/query_*.py
# To:   NLQtoDataInsight/ (project root)
current_file = Path(__file__)
codes_dir = current_file.parent.parent.parent  # ai_data
project_root = codes_dir.parent.parent         # NLQtoDataInsight
sys.path.insert(0, str(project_root))

# land/ocean mask helper (aligned to LLC2160 grid subset)
mask_module_path = os.path.abspath(os.path.join(project_root, 'LLC2160_land_ocean_mask'))
if mask_module_path not in sys.path:
    sys.path.insert(0, mask_module_path)
from utils import compute_land_ocean_mask3d

# -------------------------
# Configuration (from Pre-Analyzer)
# -------------------------
quality = 0  # Use this exact value
x_range = [2832, 3312]  # inclusive-exclusive indices (width = 480)
y_range = [4090, 4575]  # inclusive-exclusive indices (height = 485)
z_range = [0, 1]        # surface layer only
lat_range = [5.036710739135742, 22.983346939086914]   # metadata bounds
lon_range = [80.02083587646484, 99.97916412353516]     # metadata bounds
target_timestep = 6150  # single hourly snapshot
time_range_triplet = [target_timestep, target_timestep, 1]

# Dataset temporal info for human-readable time
dataset_start = datetime.strptime("2020-01-20", "%Y-%m-%d")
human_time = (dataset_start + timedelta(hours=target_timestep)).strftime("%Y-%m-%d %H:%M:%S UTC")

# Salinity dataset (OpenVisus IDX)
url = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc2160_salt/salt_llc2160_x_y_depth.idx"

# Output cache path
out_npz_path = "/Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app/ai_data/data_cache/dyamond_llc2160/data_20251201_163734.npz"

def ensure_yx(arr, ny_expected, nx_expected):
    """
    Ensure a 2D array is in (ny, nx) orientation.
    If arr is (nx, ny) it will be transposed.
    """
    a = np.squeeze(arr)
    if a.ndim != 2:
        raise RuntimeError(f"Expected 2D array after squeeze, got shape {a.shape}")
    ny, nx = a.shape
    if ny == ny_expected and nx == nx_expected:
        return a
    if ny == nx_expected and nx == ny_expected:
        return a.T
    # As fallback, try to reshape if total size matches
    if a.size == ny_expected * nx_expected:
        return a.reshape(ny_expected, nx_expected)
    raise RuntimeError(f"Cannot match array shape {a.shape} to expected (ny, nx)=({ny_expected}, {nx_expected})")

try:
    # Land/ocean mask for the exact subset (surface only)
    mask3d = compute_land_ocean_mask3d(
        time=0,  # static in time
        quality=quality,
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        use_surface_only=True
    )
    ocean_mask = (np.squeeze(mask3d) == 1)

    # Load dataset and read single timestep
    ds = ovp.LoadDataset(url)
    # Sanity: available timesteps (informative)
    _ = len(ds.db.getTimesteps())

    # Read salinity for the chosen time and subset
    raw = ds.db.read(
        time=target_timestep,
        x=[x_range[0], x_range[1]],
        y=[y_range[0], y_range[1]],
        z=[z_range[0], z_range[1]],
        quality=quality
    )

    # Expected sizes
    nx_expected = x_range[1] - x_range[0]  # 480
    ny_expected = y_range[1] - y_range[0]  # 485

    # Make sure salinity is (ny, nx)
    S = ensure_yx(raw, ny_expected, nx_expected).astype(np.float32)

    # Ensure ocean_mask matches (ny, nx)
    ocean_mask = ensure_yx(ocean_mask, ny_expected, nx_expected)

    # Apply ocean mask (land -> NaN)
    S_masked = np.where(ocean_mask, S, np.nan).astype(np.float32)

    # Build lat/lon vectors across the subset extent for plotting/metrics (approximation)
    lon_min, lon_max = lon_range
    lat_min, lat_max = lat_range
    nx = S_masked.shape[1]
    ny = S_masked.shape[0]
    lon_vec = np.linspace(lon_min, lon_max, nx, dtype=np.float32)
    lat_vec = np.linspace(lat_min, lat_max, ny, dtype=np.float32)

    # Compute grid spacing in meters (approximate using central latitude)
    # Note: LLC2160 is not rectilinear in lat/lon, but this provides a reasonable local metric approximation.
    if nx > 1:
        dx_deg = float(lon_max - lon_min) / float(nx - 1)
    else:
        dx_deg = 0.0
    if ny > 1:
        dy_deg = float(lat_max - lat_min) / float(ny - 1)
    else:
        dy_deg = 0.0

    lat_center = 0.5 * (lat_min + lat_max)
    meters_per_deg_lon = 111320.0 * np.cos(np.deg2rad(lat_center))
    meters_per_deg_lat = 110574.0
    dx_m = dx_deg * meters_per_deg_lon
    dy_m = dy_deg * meters_per_deg_lat

    # Guard against zero spacing (degenerate case)
    dx_m = dx_m if dx_m > 0 else 1.0
    dy_m = dy_m if dy_m > 0 else 1.0

    # Horizontal gradients (units: g/kg per meter ~ PSU per meter)
    dSdy, dSdx = np.gradient(S_masked, dy_m, dx_m)  # order: (y, x) spacings
    # Magnitude in PSU/km
    grad_mag_psu_per_km = (np.sqrt(dSdx**2 + dSdy**2) * 1e3).astype(np.float32)
    dSdx = dSdx.astype(np.float32)
    dSdy = dSdy.astype(np.float32)

    # Stats for reporting
    def nanstats(a):
        return {
            "mean": float(np.nanmean(a)),
            "min": float(np.nanmin(a)),
            "max": float(np.nanmax(a)),
            "std": float(np.nanstd(a))
        }

    results = [{
        "timestep": int(target_timestep),
        "human_time": human_time,
        "salinity_stats": nanstats(S_masked),
        "grad_mag_psu_per_km_stats": nanstats(grad_mag_psu_per_km)
    }]

    data_points_processed = int(S_masked.size)

    # Save NPZ cache for plotting
    np.savez(
        out_npz_path,
        variable_names=np.array(["salinity"], dtype=object),
        salinity=S_masked,
        grad_mag_psu_per_km=grad_mag_psu_per_km,
        dSdx=dSdx,
        dSdy=dSdy,
        ocean_mask=ocean_mask.astype(np.uint8),
        lon_vec=lon_vec,
        lat_vec=lat_vec,
        x_range=np.array(x_range, dtype=np.int32),
        y_range=np.array(y_range, dtype=np.int32),
        z_range=np.array(z_range, dtype=np.int32),
        time_range=np.array(time_range_triplet, dtype=np.int32),
        timestep=np.array([target_timestep], dtype=np.int32),
        time_iso=np.array([human_time], dtype=object),
        lat_range=np.array(lat_range, dtype=np.float32),
        lon_range=np.array(lon_range, dtype=np.float32),
        quality_level=np.array(quality, dtype=np.int32)
    )

    # Output JSON summary
    summary = {
        "status": "success",
        "variable_name(s)": "salinity",
        "strategy": "Read near-surface salinity at full resolution (Q=0) for Bay of Bengal subset at a single timestep (6150 ~ 2020-10-02 06:00 UTC). Applied ocean mask and computed horizontal gradients and magnitude (PSU/km) using approximate metric from lat/lon bounds.",
        "data_points_processed": data_points_processed,
        "x_range": x_range,
        "y_range": y_range,
        "z_range": z_range,
        "time_range": time_range_triplet,
        "timestep": target_timestep,
        "time_human": human_time,
        "lat_range": lat_range,
        "lon_range": lon_range,
        "quality_level": quality,
        "salinity_stats": results[0]["salinity_stats"],
        "grad_mag_psu_per_km_stats": results[0]["grad_mag_psu_per_km_stats"],
        "cache_path": out_npz_path
    }
    print(json.dumps(summary))

except Exception as e:
    print(json.dumps({"error": str(e)}))