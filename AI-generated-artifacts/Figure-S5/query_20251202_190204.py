import openvisuspy as ovp
import numpy as np
import json
from datetime import datetime, timedelta
import sys, os
from pathlib import Path

# ------------------------------------------------------------
# Project path setup for optional land/ocean mask utility
# ------------------------------------------------------------
try:
    current_file = Path(__file__)  # query_*.py
except NameError:
    # Fallback if __file__ is not defined (interactive)
    current_file = Path.cwd() / "query_generated.py"

codes_dir = current_file.parent.parent.parent  # ai_data/codes/dyamond_llc2160 -> ai_data/codes
project_root = codes_dir.parent.parent  # -> NLQtoDataInsight/
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

mask_module_path = os.path.abspath(os.path.join(project_root, 'LLC2160_land_ocean_mask'))
if mask_module_path not in sys.path:
    sys.path.insert(0, mask_module_path)

# Try to import mask utility; fall back to "all ocean" if unavailable
compute_mask_available = False
try:
    from utils import compute_land_ocean_mask3d
    compute_mask_available = True
except Exception:
    compute_mask_available = False

# ------------------------------------------------------------
# Configuration from pre-analyzer (USE EXACTLY AS GIVEN)
# ------------------------------------------------------------
quality = 0
x_range = [1392, 3792]
y_range = [2457, 4779]
z_range = [0, 1]
lat_range = [-49.9940948486, 29.9722290039]
lon_range = [20.0208339691, 119.9791641235]

# Dataset and variable
temperature_url = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc2160_theta/llc2160_theta.idx"
variable_name = "temperature"
variable_unit = "degrees Celsius"

# Temporal strategy (USE EXACTLY AS GIVEN)
dataset_start = datetime.strptime("2020-01-20", "%Y-%m-%d")
may_target = datetime.strptime("2020-05-15 12:00", "%Y-%m-%d %H:%M")
nov_target = datetime.strptime("2020-11-15 12:00", "%Y-%m-%d %H:%M")
timestep_may = 2796
timestep_nov = 7212
timesteps = [timestep_may, timestep_nov]
datestrs = {
    timestep_may: may_target.strftime("%Y-%m-%d %H:%M UTC"),
    timestep_nov: nov_target.strftime("%Y-%m-%d %H:%M UTC")
}

# Output cache
cache_path = "/Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app/ai_data/data_cache/dyamond_llc2160/data_20251202_190204.npz"

def extract_surface_2d(arr3d, depth_len):
    """
    Given a 3D array with one axis equal to depth_len (2 here for z=[0,1]),
    return the z=0 surface as a 2D array by selecting along that axis.
    """
    if arr3d.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr3d.shape}")
    shape = arr3d.shape
    # Identify depth axis by its length
    depth_axis_candidates = [i for i, s in enumerate(shape) if s == depth_len]
    if not depth_axis_candidates:
        # Some OpenVisus builds may squeeze dimensions when requesting z=[0,1].
        # If depth not identifiable, assume conventional order [Z, Y, X] and try axis 0.
        depth_axis = 0
    else:
        depth_axis = depth_axis_candidates[0]
    # Select surface level index 0 along depth axis
    surface2d = np.take(arr3d, indices=0, axis=depth_axis)
    # Ensure result is 2D
    if surface2d.ndim != 2:
        # In some cases, the read could have transposed order; try to squeeze
        surface2d = np.squeeze(surface2d)
    if surface2d.ndim != 2:
        raise ValueError(f"Could not reduce to 2D surface. Got shape after selection: {surface2d.shape}")
    return surface2d

def get_mask2d(x_range, y_range, z_range, quality):
    """
    Returns a 2D boolean ocean mask aligned with the temperature subset.
    If the mask utility is unavailable, returns an all-True mask.
    """
    if compute_mask_available:
        try:
            mask3d = compute_land_ocean_mask3d(
                time=0,
                quality=quality,
                x_range=x_range,
                y_range=y_range,
                z_range=z_range,
                use_surface_only=True  # efficient; same boundary at all depths
            )
            mask_arr = np.array(mask3d)
            # Convert to boolean ocean mask (1=ocean, 0=land)
            # Reduce to 2D
            if mask_arr.ndim == 3:
                # depth might be axis with size 1 or 2; select first slice then squeeze
                depth_axis_cands = [i for i, s in enumerate(mask_arr.shape) if s in (1, 2)]
                if depth_axis_cands:
                    mask2d = np.take(mask_arr, indices=0, axis=depth_axis_cands[0]).squeeze()
                else:
                    mask2d = mask_arr.squeeze()
            elif mask_arr.ndim == 2:
                mask2d = mask_arr
            else:
                mask2d = np.squeeze(mask_arr)
            mask2d = (mask2d == 1)
            return mask2d
        except Exception:
            pass
    # Fallback: assume all ocean (no masking)
    # Determine expected 2D shape from data read of minimal test (we estimate from ranges)
    nx = x_range[1] - x_range[0] + 1
    ny = y_range[1] - y_range[0] + 1
    return np.ones((ny, nx), dtype=bool)

def nanstats(arr):
    return {
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr))
    }

def nanpercentiles(arr, percentiles=(5, 25, 50, 75, 95)):
    pts = np.nanpercentile(arr, percentiles)
    return {f"p{p}": float(v) for p, v in zip(percentiles, pts)}

def main():
    status = "success"
    message = ""
    try:
        # Load dataset
        ds = ovp.LoadDataset(temperature_url)

        # Prepare mask
        mask2d = get_mask2d(x_range, y_range, z_range, quality)
        mask2d = mask2d.astype(bool)

        # Read data for two timesteps
        data_points_processed = 0
        results = []

        temp_may = None
        temp_nov = None

        for t in timesteps:
            data3d = ds.db.read(
                time=int(t),
                x=[x_range[0], x_range[1]],
                y=[y_range[0], y_range[1]],
                z=[z_range[0], z_range[1]],
                quality=quality
            )
            # Ensure numpy array
            data3d = np.array(data3d)
            # Surface (z=0)
            surface2d = extract_surface_2d(data3d, depth_len=(z_range[1] - z_range[0] + 1))
            # Align mask shape if necessary (transpose if needed)
            if mask2d.shape != surface2d.shape:
                # Attempt simple transpose if that fixes shape
                if mask2d.T.shape == surface2d.shape:
                    mask_use = mask2d.T
                else:
                    # As last resort, broadcast/crop to matching shape
                    ny, nx = surface2d.shape
                    mask_use = np.ones((ny, nx), dtype=bool)
            else:
                mask_use = mask2d

            surf_masked = np.where(mask_use, surface2d, np.nan)

            if t == timestep_may:
                temp_may = surf_masked.astype(np.float32)
            elif t == timestep_nov:
                temp_nov = surf_masked.astype(np.float32)

            data_points_processed += surface2d.size

            stats = nanstats(surf_masked)
            percs = nanpercentiles(surf_masked)
            results.append({
                "timestep": int(t),
                "date": datestrs[t],
                "stats": stats,
                "percentiles": percs
            })

        # Compute anomaly (Nov - May)
        if temp_may is None or temp_nov is None:
            raise RuntimeError("Failed to read one or both timesteps.")
        # Align shapes if needed
        if temp_may.shape != temp_nov.shape:
            # Try transpose
            if temp_may.T.shape == temp_nov.shape:
                temp_may = temp_may.T
            elif temp_nov.T.shape == temp_may.shape:
                temp_nov = temp_nov.T
            else:
                # fallback: crop to min common shape
                ny = min(temp_may.shape[0], temp_nov.shape[0])
                nx = min(temp_may.shape[1], temp_nov.shape[1])
                temp_may = temp_may[:ny, :nx]
                temp_nov = temp_nov[:ny, :nx]
                # also crop mask to min shape
                mask2d = mask2d[:ny, :nx] if mask2d.shape[:2] != (ny, nx) else mask2d

        anomaly = (temp_nov - temp_may).astype(np.float32)

        # Statistics and histograms
        stats_may = nanstats(temp_may)
        stats_nov = nanstats(temp_nov)
        stats_anom = nanstats(anomaly)

        percs_may = nanpercentiles(temp_may)
        percs_nov = nanpercentiles(temp_nov)
        percs_anom = nanpercentiles(anomaly)

        # Histograms: use common bin edges for May/Nov (to compare), and symmetric bins for anomaly
        finite_may = temp_may[np.isfinite(temp_may)]
        finite_nov = temp_nov[np.isfinite(temp_nov)]
        finite_anom = anomaly[np.isfinite(anomaly)]
        overall_min = float(np.nanmin([finite_may.min() if finite_may.size else np.nan,
                                       finite_nov.min() if finite_nov.size else np.nan]))
        overall_max = float(np.nanmax([finite_may.max() if finite_may.size else np.nan,
                                       finite_nov.max() if finite_nov.size else np.nan]))
        if not np.isfinite(overall_min) or not np.isfinite(overall_max) or overall_min == overall_max:
            # fallback to generic range
            overall_min, overall_max = -2.0, 35.0

        bins_val = np.linspace(overall_min, overall_max, 51, dtype=np.float32)
        counts_may, edges_val = np.histogram(finite_may, bins=bins_val)
        counts_nov, _ = np.histogram(finite_nov, bins=bins_val)

        max_abs_anom = float(np.nanmax(np.abs(finite_anom))) if finite_anom.size else 1.0
        bins_anom = np.linspace(-max_abs_anom, max_abs_anom, 51, dtype=np.float32)
        counts_anom, edges_anom = np.histogram(finite_anom, bins=bins_anom)

        # Prepare save dict for .npz
        save_dict = {
            "variable_name": np.array(variable_name),
            "variable_unit": np.array(variable_unit),
            "dataset_url": np.array(temperature_url),
            "quality_level": np.array(quality, dtype=np.int32),
            "x_range": np.array(x_range, dtype=np.int32),
            "y_range": np.array(y_range, dtype=np.int32),
            "z_range": np.array(z_range, dtype=np.int32),
            "lat_range": np.array(lat_range, dtype=np.float64),
            "lon_range": np.array(lon_range, dtype=np.float64),
            "timestep_may": np.array(timestep_may, dtype=np.int32),
            "timestep_nov": np.array(timestep_nov, dtype=np.int32),
            "date_may": np.array(datestrs[timestep_may]),
            "date_nov": np.array(datestrs[timestep_nov]),
            "temp_may": temp_may,
            "temp_nov": temp_nov,
            "anomaly": anomaly,
            "ocean_mask": mask2d.astype(np.uint8),
            # Stats
            "stats_may": np.array([stats_may["min"], stats_may["max"], stats_may["mean"], stats_may["std"]], dtype=np.float32),
            "stats_nov": np.array([stats_nov["min"], stats_nov["max"], stats_nov["mean"], stats_nov["std"]], dtype=np.float32),
            "stats_anom": np.array([stats_anom["min"], stats_anom["max"], stats_anom["mean"], stats_anom["std"]], dtype=np.float32),
            "percentiles_may": np.array([percs_may["p5"], percs_may["p25"], percs_may["p50"], percs_may["p75"], percs_may["p95"]], dtype=np.float32),
            "percentiles_nov": np.array([percs_nov["p5"], percs_nov["p25"], percs_nov["p50"], percs_nov["p75"], percs_nov["p95"]], dtype=np.float32),
            "percentiles_anom": np.array([percs_anom["p5"], percs_anom["p25"], percs_anom["p50"], percs_anom["p75"], percs_anom["p95"]], dtype=np.float32),
            # Histograms
            "hist_edges_value": edges_val.astype(np.float32),
            "hist_counts_may": counts_may.astype(np.int64),
            "hist_counts_nov": counts_nov.astype(np.int64),
            "hist_edges_anom": edges_anom.astype(np.float32),
            "hist_counts_anom": counts_anom.astype(np.int64)
        }

        # Ensure output directory exists
        out_dir = os.path.dirname(cache_path)
        os.makedirs(out_dir, exist_ok=True)

        # Save for plotting
        np.savez(cache_path, **save_dict)

        # Prepare JSON summary
        json_out = {
            "status": status,
            "strategy": "Read temperature at full resolution (Q=0) for Indian Ocean subset; timesteps: 2020-05-15 12:00 UTC (t=2796) and 2020-11-15 12:00 UTC (t=7212); apply ocean mask; compute stats, percentiles, histograms; save arrays and metadata.",
            "variable": variable_name,
            "unit": variable_unit,
            "dataset_url": temperature_url,
            "x_range": x_range,
            "y_range": y_range,
            "z_range": z_range,
            "lat_range": lat_range,
            "lon_range": lon_range,
            "timestep_may": timestep_may,
            "timestep_nov": timestep_nov,
            "date_may": datestrs[timestep_may],
            "date_nov": datestrs[timestep_nov],
            "data_points_processed": int(data_points_processed),
            "cache_path": cache_path,
            "stats": {
                "may": stats_may,
                "nov": stats_nov,
                "anomaly": stats_anom
            },
            "percentiles": {
                "may": percs_may,
                "nov": percs_nov,
                "anomaly": percs_anom
            }
        }
        print(json.dumps(json_out))
    except Exception as e:
        status = "error"
        message = str(e)
        print(json.dumps({"status": status, "error": message}))

if __name__ == "__main__":
    main()