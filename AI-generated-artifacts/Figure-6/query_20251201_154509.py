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
# mask module path 
mask_module_path = os.path.abspath(os.path.join(project_root, 'LLC2160_land_ocean_mask'))
if mask_module_path not in sys.path:
    sys.path.insert(0, mask_module_path)
from utils import compute_land_ocean_mask3d

def extract_surface(arr):
    # Try to robustly get 2D surface slice from possible 3D arrays
    if arr.ndim == 3:
        # Heuristic: if first dim is z (1 or 2), take index 0
        if arr.shape[0] in (1, 2):
            return arr[0, :, :]
        # If last dim looks like z
        if arr.shape[-1] in (1, 2):
            return arr[:, :, 0]
        # Fallback: assume first dim is z
        return arr[0, :, :]
    return arr

def apply_ocean_mask(field2d, ocean_mask):
    # Broadcast-safe masking to set land to NaN
    if ocean_mask is None:
        return field2d
    try:
        return np.where(ocean_mask, field2d, np.nan)
    except ValueError:
        # Try to reshape if needed
        if ocean_mask.ndim == 3 and ocean_mask.shape[0] == 1:
            om2d = ocean_mask[0, ...]
            return np.where(om2d, field2d, np.nan)
        raise

def compute_diagnostics(u2d, v2d):
    # Compute gradients in index-space (grid-step units)
    # du/dx, du/dy, dv/dx, dv/dy
    du_dy, du_dx = np.gradient(u2d)  # gradient returns [d/dy, d/dx] for 2D
    dv_dy, dv_dx = np.gradient(v2d)
    # Relative vorticity (index-units): zeta = dv/dx - du/dy
    zeta = dv_dx - du_dy
    # Strain components
    s_n = du_dx - dv_dy
    s_s = dv_dx + du_dy
    # Okubo–Weiss parameter
    ow = s_n**2 + s_s**2 - zeta**2
    return zeta.astype(np.float32), ow.astype(np.float32)

def main():
    # Fixed configuration (from pre-analyzer; use exactly)
    quality = -6  # Use this exact value
    x_range = [3840, 4392]  # end-exclusive semantics expected by OpenVisus (matches pre-analysis counts)
    y_range = [4547, 4966]
    z_range = [0, 1]  # We'll read and then take surface (z=0)
    lat_range = [22.02800178527832, 35.98868179321289]  # For metadata
    lon_range = [122.02083587646484, 144.9791717529297]

    # Dataset URLs for velocity components
    u_url = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc2160_arco/visus.idx"
    v_url = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc2160_v/v_llc2160_x_y_depth.idx"

    # Output cache path
    out_npz = '/Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app/ai_data/data_cache/dyamond_llc2160/data_20251201_154509.npz'

    # Temporal mapping (dataset time units: hours; dataset start: 2020-01-20)
    dataset_start = datetime.strptime("2020-01-20", "%Y-%m-%d")
    dataset_end = datetime.strptime("2021-03-26", "%Y-%m-%d")

    user_start = datetime.strptime("2020-01-01", "%Y-%m-%d")
    user_end = datetime.strptime("2020-03-31", "%Y-%m-%d")

    # Compute hourly indices, clamp to dataset
    t0 = int(max(0, (user_start - dataset_start).total_seconds() // 3600))  # clamps to 0
    t1 = int(min(int((user_end - dataset_start).total_seconds() // 3600), 10366 - 1))
    # We want daily snapshots at 00:00 UTC; with dataset_start aligned to 00:00, stride=24
    timestep_step = 24

    # Ensure inclusive end and alignment to 24-hour grid
    # If t0 not divisible by 24 (should be 0), align up
    if t0 % 24 != 0:
        t0 = t0 + (24 - (t0 % 24))
    if t1 % 24 != 0:
        t1 = t1 - (t1 % 24)
    timesteps = list(range(t0, t1 + 1, timestep_step))

    # Build date strings for each timestep for convenience
    dates = [(dataset_start + timedelta(hours=int(t))).strftime("%Y-%m-%d %H:%M:%S") for t in timesteps]

    # Land/ocean mask for the region (surface only for efficiency)
    mask3d = compute_land_ocean_mask3d(
        time=0,  # Land/ocean boundary doesn't change with time
        quality=quality,  # Use same quality as data for alignment
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        use_surface_only=True  # Efficient: land/ocean same at all depths
    )
    # Convert to 2D ocean mask
    if mask3d.ndim == 3 and mask3d.shape[0] == 1:
        ocean_mask = (mask3d[0, ...] == 1)
    elif mask3d.ndim == 2:
        ocean_mask = (mask3d == 1)
    else:
        # Fallback: try first slice
        ocean_mask = (extract_surface(mask3d) == 1)

    try:
        # Load datasets
        ds_u = ovp.LoadDataset(u_url)
        ds_v = ovp.LoadDataset(v_url)

        # Sanity on available timesteps
        # If available, use dataset-provided timesteps to clamp
        try:
            length_of_timesteps_u = len(ds_u.db.getTimesteps())
            length_of_timesteps_v = len(ds_v.db.getTimesteps())
            max_steps = min(length_of_timesteps_u, length_of_timesteps_v)
            timesteps = [t for t in timesteps if t < max_steps]
        except Exception:
            pass  # proceed with precomputed indices

        # Pre-allocate after first read to match actual array shape
        u_all = None
        v_all = None
        zeta_all = None
        ow_all = None

        data_points_processed = 0
        results = []

        x0, x1 = x_range[0], x_range[1]
        y0, y1 = y_range[0], y_range[1]
        z0, z1 = z_range[0], z_range[1]  # We'll still read [0,1] and select surface

        for idx, t in enumerate(timesteps):
            # Read u and v (near-surface slice will be extracted)
            u_raw = ds_u.db.read(
                time=int(t),
                x=[x0, x1],
                y=[y0, y1],
                z=[z0, z1],
                quality=quality
            )
            v_raw = ds_v.db.read(
                time=int(t),
                x=[x0, x1],
                y=[y0, y1],
                z=[z0, z1],
                quality=quality
            )

            # Extract 2D surface
            u2d = extract_surface(u_raw).astype(np.float32)
            v2d = extract_surface(v_raw).astype(np.float32)

            # Initialize containers on first loop based on actual shapes
            if u_all is None:
                ny, nx = u2d.shape
                nt = len(timesteps)
                u_all = np.full((nt, ny, nx), np.nan, dtype=np.float32)
                v_all = np.full((nt, ny, nx), np.nan, dtype=np.float32)
                zeta_all = np.full((nt, ny, nx), np.nan, dtype=np.float32)
                ow_all = np.full((nt, ny, nx), np.nan, dtype=np.float32)
                # Conform ocean_mask shape
                if ocean_mask.shape != (ny, nx):
                    ocean_mask_local = extract_surface(mask3d)
                    if ocean_mask_local.shape != (ny, nx):
                        # As a last resort, try to slice/resize (not ideal). If mismatch, disable mask.
                        ocean_mask_local = None
                    else:
                        ocean_mask = (ocean_mask_local == 1)

            # Apply ocean mask
            if ocean_mask is not None and ocean_mask.shape == u2d.shape:
                u2d = np.where(ocean_mask, u2d, np.nan)
                v2d = np.where(ocean_mask, v2d, np.nan)

            # Compute diagnostics
            zeta2d, ow2d = compute_diagnostics(u2d, v2d)

            # Store
            u_all[idx, :, :] = u2d
            v_all[idx, :, :] = v2d
            zeta_all[idx, :, :] = zeta2d
            ow_all[idx, :, :] = ow2d

            # Stats (nan-aware)
            speed = np.sqrt(u2d**2 + v2d**2)
            your_finding = {
                "timestep": int(t),
                "date_utc": dates[idx],
                "mean_speed": float(np.nanmean(speed)),
                "min_speed": float(np.nanmin(speed)),
                "max_speed": float(np.nanmax(speed)),
                "std_speed": float(np.nanstd(speed)),
                "mean_vorticity": float(np.nanmean(zeta2d)),
                "min_vorticity": float(np.nanmin(zeta2d)),
                "max_vorticity": float(np.nanmax(zeta2d)),
                "mean_okuboweiss": float(np.nanmean(ow2d)),
                "min_okuboweiss": float(np.nanmin(ow2d)),
                "max_okuboweiss": float(np.nanmax(ow2d))
            }
            results.append(your_finding)

            # Update processed points (u and v counts)
            data_points_processed += (u2d.size + v2d.size)

        # Save to NPZ (store results as JSON string to avoid pickle)
        results_json = json.dumps(results)

        np.savez(
            out_npz,
            variable_names=np.array(["u", "v", "zeta", "okubo_weiss"], dtype=object),
            x_range=np.array(x_range, dtype=np.int64),
            y_range=np.array(y_range, dtype=np.int64),
            z_range=np.array(z_range, dtype=np.int64),
            time_range=np.array([t0, t1, timestep_step], dtype=np.int64),
            timesteps=np.array(timesteps, dtype=np.int64),
            dates=np.array(dates, dtype=object),
            lat_range=np.array(lat_range, dtype=np.float64),
            lon_range=np.array(lon_range, dtype=np.float64),
            quality_level=np.array(quality, dtype=np.int32),
            ocean_mask=ocean_mask.astype(np.bool_),
            u=u_all,
            v=v_all,
            zeta=zeta_all,
            okubo_weiss=ow_all,
            results_json=np.array(results_json, dtype=object)
        )

        # Output summary JSON to stdout
        print(json.dumps({
            "status": "success",
            "variable_name(s)": "velocity(u,v)",
            "strategy": "Daily sampling at 00:00 UTC from 2020-01-20 to 2020-03-31 (inclusive), surface slice (z=0) from specified Kuroshio subdomain; compute and cache relative vorticity and Okubo–Weiss at quality -6.",
            "data_points_processed": int(data_points_processed),
            "x_range": x_range,
            "y_range": y_range,
            "z_range": z_range,
            "time_range": [int(t0), int(t1), int(timestep_step)],
            "timesteps_count": len(timesteps),
            "lat_range": lat_range,
            "lon_range": lon_range,
            "quality_level": quality,
            "cache_path": out_npz,
            "results": results[:5] + ([{"...": f"{len(results)-5} more"}] if len(results) > 5 else [])
        }))

    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()