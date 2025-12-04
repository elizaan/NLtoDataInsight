import openvisuspy as ovp
import numpy as np
import json
from datetime import datetime
import sys, os
from pathlib import Path

# -------------------------
# Project paths for helpers
# -------------------------
current_file = Path(__file__)  # query_*.py
codes_dir = current_file.parent.parent.parent  # ai_data
project_root = codes_dir.parent.parent  # NLQtoDataInsight
sys.path.insert(0, str(project_root))

mask_module_path = os.path.abspath(os.path.join(project_root, 'LLC2160_land_ocean_mask'))
if mask_module_path not in sys.path:
    sys.path.insert(0, mask_module_path)

try:
    from utils import compute_land_ocean_mask3d
    HAS_MASK_UTIL = True
except Exception:
    HAS_MASK_UTIL = False

def clamp(v, lo, hi):
    return max(lo, min(v, hi))

def to_numpy(arr):
    try:
        return np.asarray(arr)
    except Exception:
        return np.array(arr)

def main():
    # Fixed configuration from PRE-ANALYZER (use exactly as provided)
    quality = 0
    x_range = [7200, 7608]
    y_range = [4436, 4810]
    z_range = [0, 90]
    lat_range = [18.033620834350586, 30.99648666381836]
    lon_range = [-97.97916412353516, -81.02083587646484]

    # Output cache path
    out_npz = '/Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app/ai_data/data_cache/dyamond_llc2160/data_20251202_202907.npz'

    # Dataset temporal info
    dataset_start = datetime.strptime("2020-01-20", "%Y-%m-%d")
    dataset_end = datetime.strptime("2021-03-26", "%Y-%m-%d")
    total_timesteps = 10366  # hours
    
    # Target snapshot (from PRE-ANALYZER)
    target_time = datetime.strptime("2020-08-01 00:00", "%Y-%m-%d %H:%M")
    # Compute timestep index
    hours_from_start = int((target_time - dataset_start).total_seconds() / 3600)
    timestep_index = clamp(hours_from_start, 0, total_timesteps - 1)

    # Velocity component datasets (OpenVisus .idx)
    urls = {
        "u": "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc2160_arco/visus.idx",
        "v": "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc2160_v/v_llc2160_x_y_depth.idx",
        "w": "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc2160_w/llc2160_w.idx",
    }

    try:
        # Load datasets
        ds_u = ovp.LoadDataset(urls["u"])
        ds_v = ovp.LoadDataset(urls["v"])
        ds_w = ovp.LoadDataset(urls["w"])

        # Determine valid time index using ds_u as reference
        len_timesteps_u = len(ds_u.db.getTimesteps())
        t = clamp(timestep_index, 0, max(0, len_timesteps_u - 1))

        # Land/ocean mask (constant in time); use helper if available
        if HAS_MASK_UTIL:
            mask3d = compute_land_ocean_mask3d(
                time=0,
                quality=quality,
                x_range=x_range,
                y_range=y_range,
                z_range=z_range,
                use_surface_only=True
            )
            ocean_mask = (mask3d == 1)
        else:
            ocean_mask = None

        # Read u, v, w at single timestep
        u_data = ds_u.db.read(time=t, x=[x_range[0], x_range[1]], y=[y_range[0], y_range[1]], z=[z_range[0], z_range[1]], quality=quality)
        v_data = ds_v.db.read(time=t, x=[x_range[0], x_range[1]], y=[y_range[0], y_range[1]], z=[z_range[0], z_range[1]], quality=quality)
        w_data = ds_w.db.read(time=t, x=[x_range[0], x_range[1]], y=[y_range[0], y_range[1]], z=[z_range[0], z_range[1]], quality=quality)

        u = to_numpy(u_data).astype(np.float32, copy=False)
        v = to_numpy(v_data).astype(np.float32, copy=False)
        w = to_numpy(w_data).astype(np.float32, copy=False)

        # Ensure shapes are consistent
        if not (u.shape == v.shape == w.shape):
            raise RuntimeError(f"Shape mismatch among components: u{u.shape}, v{v.shape}, w{w.shape}")

        # Apply ocean mask
        if ocean_mask is not None:
            if ocean_mask.shape != u.shape:
                # If mask is 2D repeated over depth, broadcast
                try:
                    ocean_mask_b = np.broadcast_to(ocean_mask, u.shape)
                except Exception:
                    raise RuntimeError(f"Ocean mask shape {ocean_mask.shape} not compatible with data shape {u.shape}")
            else:
                ocean_mask_b = ocean_mask
            u = np.where(ocean_mask_b, u, np.nan)
            v = np.where(ocean_mask_b, v, np.nan)
            w = np.where(ocean_mask_b, w, np.nan)

        # Compute velocity magnitude
        velmag = np.sqrt(u*u + v*v + w*w).astype(np.float32, copy=False)

        # Free memory for components
        del u, v, w

        # Stats overall (ignore NaNs from land)
        overall_min = np.nanmin(velmag)
        overall_max = np.nanmax(velmag)
        overall_mean = float(np.nanmean(velmag))
        overall_std = float(np.nanstd(velmag))

        # Per-depth stats
        # Assume velmag shape is (Z, Y, X)
        if velmag.ndim != 3:
            raise RuntimeError(f"Unexpected data rank for velocity magnitude: {velmag.ndim}, expected 3")
        nz, ny, nx = velmag.shape

        per_depth_min = np.full(nz, np.nan, dtype=np.float32)
        per_depth_max = np.full(nz, np.nan, dtype=np.float32)
        per_depth_mean = np.full(nz, np.nan, dtype=np.float32)
        per_depth_std = np.full(nz, np.nan, dtype=np.float32)

        # Percentiles per depth (5th,25th,50th,75th,95th)
        pcts = [5, 25, 50, 75, 95]
        per_depth_p = np.full((nz, len(pcts)), np.nan, dtype=np.float32)

        for k in range(nz):
            slice_k = velmag[k, :, :]
            per_depth_min[k] = np.nanmin(slice_k)
            per_depth_max[k] = np.nanmax(slice_k)
            per_depth_mean[k] = np.nanmean(slice_k)
            per_depth_std[k] = np.nanstd(slice_k)
            try:
                per_depth_p[k, :] = np.nanpercentile(slice_k, pcts)
            except Exception:
                # If all-NaN (unlikely offshore), keep NaNs
                pass

        # Overall percentiles
        overall_percentiles = np.nanpercentile(velmag, pcts)

        # Histogram across all valid voxels
        valid_vals = velmag[np.isfinite(velmag)]
        # Choose histogram upper bound as 99.5th percentile to avoid extreme tails in binning
        hist_upper = float(np.nanpercentile(valid_vals, 99.5)) if valid_vals.size > 0 else float(overall_max)
        if hist_upper <= 0:
            hist_upper = float(overall_max)
        hist_bins = 60
        hist_counts, hist_edges = np.histogram(valid_vals, bins=hist_bins, range=(0.0, hist_upper))

        # Prepare results summary for stdout
        results = [{
            "timestep_index": int(t),
            "timestep_datetime": target_time.strftime("%Y-%m-%d %H:%M:%S"),
            "overall": {
                "min": float(overall_min),
                "max": float(overall_max),
                "mean": float(overall_mean),
                "std": float(overall_std),
                "percentiles": {str(p): float(v) for p, v in zip(pcts, overall_percentiles)}
            },
            "per_depth": {
                "min": [float(v) for v in per_depth_min],
                "max": [float(v) for v in per_depth_max],
                "mean": [float(v) for v in per_depth_mean],
                "std": [float(v) for v in per_depth_std],
                "percentiles": {str(p): [float(x) for x in per_depth_p[:, i]] for i, p in enumerate(pcts)}
            },
            "histogram": {
                "bins": int(hist_bins),
                "range": [0.0, float(hist_upper)],
                "counts": [int(c) for c in hist_counts],
                "bin_edges": [float(e) for e in hist_edges]
            }
        }]

        # Save to NPZ for plotting
        # Note: store metadata as arrays for easy loading without pickle
        np.savez(
            out_npz,
            velocity_magnitude=velmag,  # (Z,Y,X), float32
            x_range=np.array(x_range, dtype=np.int32),
            y_range=np.array(y_range, dtype=np.int32),
            z_range=np.array(z_range, dtype=np.int32),
            lat_range=np.array(lat_range, dtype=np.float64),
            lon_range=np.array(lon_range, dtype=np.float64),
            quality_level=np.array([quality], dtype=np.int32),
            timestep_index=np.array([t], dtype=np.int32),
            timestep_datetime=np.array([target_time.strftime("%Y-%m-%d %H:%M:%S")]),
            overall_min=np.array([overall_min], dtype=np.float32),
            overall_max=np.array([overall_max], dtype=np.float32),
            overall_mean=np.array([overall_mean], dtype=np.float32),
            overall_std=np.array([overall_std], dtype=np.float32),
            overall_percentiles=np.array(overall_percentiles, dtype=np.float32),
            per_depth_min=per_depth_min,
            per_depth_max=per_depth_max,
            per_depth_mean=per_depth_mean,
            per_depth_std=per_depth_std,
            per_depth_percentiles=per_depth_p,
            histogram_counts=np.array(hist_counts, dtype=np.int64),
            histogram_bin_edges=np.array(hist_edges, dtype=np.float32),
            percentiles_list=np.array(pcts, dtype=np.int32)
        )

        # Print JSON summary to stdout
        summary = {
            "status": "success",
            "variable_name(s)": ["velocity_u", "velocity_v", "velocity_w", "velocity_magnitude"],
            "strategy": "Single hourly snapshot at 2020-08-01 00:00 UTC; full-resolution (Q=0) read of Gulf subdomain; compute velocity magnitude and statistics overall and per depth; cache magnitude for 3D plotting.",
            "data_points_processed": int(velmag.size),
            "x_range": x_range,
            "y_range": y_range,
            "z_range": z_range,
            "lat_range": lat_range,
            "lon_range": lon_range,
            "quality_level": quality,
            "timestep_index": int(t),
            "timestep_datetime": target_time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": results
        }
        print(json.dumps(summary))
    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}))

if __name__ == "__main__":
    main()