import os
import sys
import json
import math
import argparse
import calendar
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np

# Optional: Only needed if available in the runtime
try:
    import openvisuspy as ovp
except Exception as _e:
    ovp = None

# Add project root to Python path (following provided template)
# From: agent6-web-app/ai_data/codes/dyamond_llc2160/query_*.py
# To:   NLQtoDataInsight/ (project root)
current_file = Path(__file__) if '__file__' in globals() else Path.cwd() / "query_generated.py"
codes_dir = current_file.parent.parent.parent  # Go up to ai_data
project_root = codes_dir.parent.parent  # Go up to NLQtoDataInsight/
sys.path.insert(0, str(project_root))

# Land/ocean mask helper
try:
    mask_module_path = os.path.abspath(os.path.join(project_root, 'LLC2160_land_ocean_mask'))
    if mask_module_path not in sys.path:
        sys.path.insert(0, mask_module_path)
    from utils import compute_land_ocean_mask3d
except Exception as _e:
    compute_land_ocean_mask3d = None

def parse_args():
    parser = argparse.ArgumentParser(description="Query DYAMOND LLC2160 SST for Gulf Stream anomalies (daily sampling, Q=-3).")
    parser.add_argument(
        "--months",
        type=str,
        default=os.environ.get("DYAMOND_MONTHS", "2020-03"),
        help="Comma-separated list of months in YYYY-MM within [2020-01..2021-03] (e.g., '2020-03,2020-04'). Default: 2020-03"
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default="/Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app/ai_data/data_cache/dyamond_llc2160/data_20251202_044455.npz",
        help="Output NPZ path for cached arrays and metadata"
    )
    return parser.parse_args()

def month_start_end(year: int, month: int):
    start = datetime(year, month, 1)
    last_day = calendar.monthrange(year, month)[1]
    end = datetime(year, month, last_day, 23, 59, 59)
    return start, end

def clamp_to_dataset(user_start_dt, user_end_dt, dataset_start_dt, dataset_end_dt):
    start_dt = max(user_start_dt, dataset_start_dt)
    end_dt = min(user_end_dt, dataset_end_dt)
    if end_dt < start_dt:
        return None, None
    return start_dt, end_dt

def datetime_to_timestep(dt, dataset_start_dt, clamp_low=0, clamp_high=10366-1):
    # Dataset time unit = hours
    hours = int((dt - dataset_start_dt).total_seconds() // 3600)
    return max(clamp_low, min(hours, clamp_high))

def main():
    args = parse_args()

    # PRE-ANALYZER SPECIFICATIONS (USE THESE DIRECTLY)
    quality = -3  # exact
    x_range = [7512, 8592]  # exact
    y_range = [4603, 5463]  # exact
    z_range = [0, 1]        # exact (we will use surface slice z=0 from the read)
    lat_range = [24.001941680908203, 49.99409484863281]  # metadata
    lon_range = [-84.97916412353516, -40.02083206176758] # metadata

    url_temperature = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc2160_theta/llc2160_theta.idx"

    # Temporal Information
    dataset_start = datetime.strptime("2020-01-20", "%Y-%m-%d")
    dataset_end = datetime.strptime("2021-03-26", "%Y-%m-%d")
    total_timesteps = 10366
    time_unit = "hours"
    stride_hours = 24  # daily sampling as per spec

    # Parse months list
    months_list = [m.strip() for m in args.months.split(",") if m.strip()]
    # Validate month format
    valid_months = []
    for m in months_list:
        try:
            dt = datetime.strptime(m, "%Y-%m")
            if dt.year < 2020 or (dt.year == 2021 and dt.month > 3) or dt.year > 2021:
                # Outside dataset general window; still allow partial overlap after clamping
                pass
            valid_months.append(m)
        except Exception:
            # Skip invalid formats
            continue
    if not valid_months:
        print(json.dumps({"error": "No valid months provided. Expected format 'YYYY-MM' within 2020-01..2021-03."}))
        return

    # Ensure output directory exists
    cache_path = Path(args.cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset and mask
    if ovp is None:
        print(json.dumps({"error": "openvisuspy is not available in this environment."}))
        return

    try:
        ds = ovp.LoadDataset(url_temperature)
    except Exception as e:
        print(json.dumps({"error": f"Failed to open dataset: {str(e)}"}))
        return

    # Build ocean mask matching our subdomain; use surface-only
    if compute_land_ocean_mask3d is None:
        # Fallback: no mask available; treat all as ocean
        ocean_mask = None
    else:
        try:
            mask3d = compute_land_ocean_mask3d(
                time=0,
                quality=quality,
                x_range=x_range,
                y_range=y_range,
                z_range=z_range,
                use_surface_only=True
            )
            # Expecting 2D mask (Y,X) when use_surface_only=True; if 3D, pick surface
            mask_arr = np.array(mask3d)
            if mask_arr.ndim == 3:
                ocean_mask = (mask_arr[0] == 1)
            else:
                ocean_mask = (mask_arr == 1)
        except Exception as e:
            ocean_mask = None  # proceed without mask
    # Prepare per-month processing
    all_months_metadata = []
    save_dict = {}
    total_points_processed = 0

    # Helper for finding daily-aligned timesteps within clamped [start,end]
    def build_daily_timesteps(t0, t1, stride=24):
        # Align to the first timestep >= t0 that maintains 24h stride
        first = ((t0 + stride - 1) // stride) * stride
        if first > t1:
            return []
        return list(range(first, t1 + 1, stride))

    # Loop over requested months
    for m in valid_months:
        y = int(m[:4]); mon = int(m[5:7])
        m_start_dt, m_end_dt = month_start_end(y, mon)
        # Clamp to dataset
        c_start_dt, c_end_dt = clamp_to_dataset(m_start_dt, m_end_dt, dataset_start, dataset_end)
        if c_start_dt is None:
            # No overlap with dataset; record and continue
            all_months_metadata.append({
                "month": m,
                "status": "no_overlap",
                "requested_start": m_start_dt.isoformat(),
                "requested_end": m_end_dt.isoformat()
            })
            continue

        t_start = datetime_to_timestep(c_start_dt, dataset_start, 0, total_timesteps - 1)
        t_end = datetime_to_timestep(c_end_dt, dataset_start, 0, total_timesteps - 1)
        # Daily stride from aligned start
        timesteps = build_daily_timesteps(t_start, t_end, stride_hours)
        if len(timesteps) == 0:
            all_months_metadata.append({
                "month": m,
                "status": "no_timesteps_after_stride",
                "clamped_start": c_start_dt.isoformat(),
                "clamped_end": c_end_dt.isoformat(),
                "t_start": t_start,
                "t_end": t_end
            })
            continue

        # Read loop
        sst_daily_list = []
        sst_stats = []
        for t in timesteps:
            try:
                data = ds.db.read(
                    time=t,
                    x=[x_range[0], x_range[1]],
                    y=[y_range[0], y_range[1]],
                    z=[z_range[0], z_range[1]],
                    quality=quality
                )
            except Exception as e:
                print(json.dumps({"error": f"Read failed at timestep {t} for month {m}: {str(e)}"}))
                return

            arr = np.array(data, dtype=np.float32)
            # Expect [Z, Y, X] if 3D
            if arr.ndim == 3:
                # Use surface layer z=0
                sst2d = arr[0, :, :]
            elif arr.ndim == 2:
                sst2d = arr
            else:
                # Unexpected shape
                print(json.dumps({"error": f"Unexpected array shape {arr.shape} at timestep {t}"}))
                return

            # Apply ocean mask
            if ocean_mask is not None:
                if ocean_mask.shape != sst2d.shape:
                    # Attempt to broadcast/clip if off-by-one; otherwise skip mask
                    try:
                        sst2d = np.where(ocean_mask, sst2d, np.nan)
                    except Exception:
                        pass
                else:
                    sst2d = np.where(ocean_mask, sst2d, np.nan)

            sst_daily_list.append(sst2d)
            # Stats for JSON summary
            sst_stats.append({
                "timestep": int(t),
                "mean": float(np.nanmean(sst2d)),
                "min": float(np.nanmin(sst2d)),
                "max": float(np.nanmax(sst2d)),
                "std": float(np.nanstd(sst2d)),
            })
            total_points_processed += int(np.prod(sst2d.shape))

        # Stack into (nt, ny, nx)
        sst_daily = np.stack(sst_daily_list, axis=0)  # float32
        # Month-mean over time
        sst_month_mean = np.nanmean(sst_daily, axis=0).astype(np.float32)

        # Transect for Hovmöller: pick central y-index across the subdomain
        y_idx_mid = (y_range[1] - y_range[0]) // 2  # index in subdomain coordinates
        # Safety clamp
        y_idx_mid = int(max(0, min(y_idx_mid, sst_daily.shape[1] - 1)))
        transect = sst_daily[:, y_idx_mid, :]  # shape (nt, nx)
        transect_month_mean = np.nanmean(transect, axis=0, keepdims=True)
        transect_anom = (transect - transect_month_mean).astype(np.float32)

        # Time series diagnostics
        # 1) Domain-mean SST (not anomaly)
        dom_mean_sst = np.nanmean(sst_daily.reshape(sst_daily.shape[0], -1), axis=1).astype(np.float32)
        # 2) Front-intensity index: area-mean of |∇T| per day (unscaled grid gradients)
        front_index = []
        for k in range(sst_daily.shape[0]):
            field = sst_daily[k]
            # Fill NaNs with overall mean to allow gradient computation
            if np.isnan(field).any():
                fill_val = np.nanmean(field)
                fld = np.nan_to_num(field, nan=fill_val)
            else:
                fld = field
            gy, gx = np.gradient(fld)  # grid units
            mag = np.sqrt(gx * gx + gy * gy)
            # Exclude originally NaN ocean_masked points from mean if possible
            if ocean_mask is not None:
                mag = np.where(ocean_mask, mag, np.nan)
            front_index.append(np.nanmean(mag))
        front_index = np.array(front_index, dtype=np.float32)

        # 3) Domain-std of anomaly per day (anomaly computed vs month mean)
        sst_anom_daily = (sst_daily - sst_month_mean[None, :, :]).astype(np.float32)
        dom_std_anom = np.nanstd(sst_anom_daily.reshape(sst_anom_daily.shape[0], -1), axis=1).astype(np.float32)

        # Human-readable times array for the selected timesteps
        t0_dt = dataset_start
        times_iso = [(t0_dt + timedelta(hours=int(t))).isoformat() for t in timesteps]

        # Save per-month arrays into save_dict using unique keys
        key_prefix = m
        save_dict[f"{key_prefix}_sst_daily"] = sst_daily  # (nt, ny, nx), float32
        save_dict[f"{key_prefix}_sst_month_mean"] = sst_month_mean  # (ny, nx), float32
        save_dict[f"{key_prefix}_transect_anom"] = transect_anom  # (nt, nx), float32
        save_dict[f"{key_prefix}_timesteps"] = np.array(timesteps, dtype=np.int32)
        save_dict[f"{key_prefix}_times_iso"] = np.array(times_iso, dtype="U32")
        save_dict[f"{key_prefix}_dom_mean_sst"] = dom_mean_sst  # (nt,), float32
        save_dict[f"{key_prefix}_front_index"] = front_index    # (nt,), float32
        save_dict[f"{key_prefix}_dom_std_anom"] = dom_std_anom  # (nt,), float32
        save_dict[f"{key_prefix}_transect_y_index"] = np.array([y_idx_mid], dtype=np.int32)

        # Month metadata
        all_months_metadata.append({
            "month": m,
            "status": "ok",
            "clamped_start": c_start_dt.isoformat(),
            "clamped_end": c_end_dt.isoformat(),
            "t_start": int(t_start),
            "t_end": int(t_end),
            "num_days": int(len(timesteps)),
            "transect_y_index": int(y_idx_mid),
            "stats": sst_stats
        })

    # Static metadata
    save_dict["x_range"] = np.array(x_range, dtype=np.int32)
    save_dict["y_range"] = np.array(y_range, dtype=np.int32)
    save_dict["z_range"] = np.array(z_range, dtype=np.int32)
    save_dict["lat_range"] = np.array(lat_range, dtype=np.float64)
    save_dict["lon_range"] = np.array(lon_range, dtype=np.float64)
    save_dict["quality_level"] = np.array([quality], dtype=np.int32)
    save_dict["months"] = np.array(valid_months, dtype="U7")
    save_dict["dataset_start_iso"] = np.array([dataset_start.isoformat()], dtype="U32")
    save_dict["dataset_end_iso"] = np.array([dataset_end.isoformat()], dtype="U32")
    save_dict["url_temperature"] = np.array([url_temperature], dtype="U256")
    save_dict["time_unit"] = np.array([time_unit], dtype="U16")
    save_dict["stride_hours"] = np.array([stride_hours], dtype=np.int32)
    save_dict["notes"] = np.array([
        "Daily sampling (stride=24h), surface layer extracted from z=[0,1] read. Gradient magnitudes are in grid units."
    ], dtype="U256")

    # Save NPZ
    try:
        np.savez_compressed(str(cache_path), **save_dict)
    except Exception as e:
        print(json.dumps({"error": f"Failed to save cache file: {str(e)}"}))
        return

    # Build JSON summary
    summary = {
        "status": "success",
        "strategy": "Daily sampling (stride=24h) at quality Q=-3 over Gulf Stream box; computed month-mean SST, anomalies (on load), transect Hovmöller anomalies, and time-series diagnostics.",
        "variable": "Temperature (surface; degrees Celsius)",
        "url": url_temperature,
        "x_range": x_range,
        "y_range": y_range,
        "z_range": z_range,
        "lat_range": lat_range,
        "lon_range": lon_range,
        "quality_level": quality,
        "months_requested": valid_months,
        "dataset_start": dataset_start.isoformat(),
        "dataset_end": dataset_end.isoformat(),
        "time_unit": time_unit,
        "temporal_stride_hours": stride_hours,
        "cache_path": str(cache_path),
        "data_points_processed_estimate": total_points_processed,
        "months_metadata": all_months_metadata
    }
    print(json.dumps(summary))

if __name__ == "__main__":
    main()