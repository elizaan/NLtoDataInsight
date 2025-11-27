# python3 -u - <<'PY' 2>&1 | tee sweep_run.log
# import importlib.util, sys
# spec = importlib.util.spec_from_file_location("accuracy_resolution","agent6-web-app/src/agents/accuracy_resolution.py")
# mod = importlib.util.module_from_spec(spec)
# sys.modules[spec.name] = mod
# spec.loader.exec_module(mod)

# res = mod.run_sweep(
#     output_csv_path='agent6-web-app/ai_data/accuracy_sweep_ref0.csv',
#     qualities=[-27,-25,-23,-21,-18,-15,-12,-9,-6,-3],
#     timesteps=[0],
#     save_maps=True,
#     ref_quality=0,
#     x_range_override=[1400,2000],
#     y_range_override=[2750,3000]
# )
# print('DONE', len(res))
# PY
# # In another terminal, follow progress:
# tail -f sweep_run.log
# # or monitor CSV rows as they append:
# tail -f agent6-web-app/ai_data/accuracy_sweep_ref0.csv

import os
import sys
import json
from time import perf_counter
from datetime import datetime
import numpy as np
import openvisuspy as ovp
from scipy.ndimage import zoom
"""
Human-readable metric & mask notes

Metrics written by this script (brief, plain English):
- rmse: Root-mean-square error between reference and test on the compared points.
    It summarizes the typical magnitude of errors (sensitive to outliers).
- bias: Mean(test - reference) over compared points. Positive bias means test is
    on average higher than the reference; negative bias means lower.
- max_abs: The single largest absolute error found (an outlier metric).
- median_abs: The median absolute error (robust central measure; 50% of errors
    are <= this value).
- avg_abs_error: The average absolute error computed over the entire reference
    grid (includes masked points as land_value substitutions). This is calculated
    as sum(|test-ref|)/N_ref_grid_points.
- pct_avg_error: The previous average expressed as a percent of the reference
    data range (100 * avg_abs_error / (ref_max - ref_min)). Useful to compare
    relative error independent of units.
- pct_within_<t>: Percentage of compared points whose absolute error is <= t
    (e.g. pct_within_0.1 = fraction of points with |error| <= 0.1). These are
    coverage statistics (not percentiles).
- valid_count: Number of grid points actually compared (non-NaN and, when a
    land/ocean mask is applied, ocean-only points). If the mask is static and
    reused, valid_count will be identical for all qualities/timesteps that use
    the same mask/region.

Why compute the mask (compute_land_ocean_mask3d) instead of always calling
apply_land_ocean_mask():
- compute_land_ocean_mask3d(...) returns a boolean/uint8 mask (shape nz,ny,nx)
    indicating ocean vs land. This is useful when you want to:
    * compute valid point indices once and reuse them for many comparisons (cheap)
    * avoid creating copies of the full data arrays just to mask out land
    * apply custom diagnostics that operate on masked vs unmasked arrays
- apply_land_ocean_mask(data, mask3d, land_value) is a convenience that
    returns a new array with land set to `land_value` (e.g. np.nan). It is
    convenient and explicit, but it produces an array copy (memory + time cost)
    and is not strictly necessary when we only need to compute statistics over
    ocean points. For the sweep we typically compute the mask once and then
    either (a) form a boolean `valid` selection (faster) or (b) call
    apply_land_ocean_mask when a masked array is easier to reason about.

Recommendation:
- If land/ocean truly does not change (it normally doesn't), compute the mask
    once at a sensible resolution (use_surface_only=True then broadcast) and
    reuse it for all qualities/timesteps. If you need the mask sampled to a
    particular grid, resample the mask with nearest-neighbor (order=0) before
    using it to avoid interpolation artifacts.

Where to look:
- Mask helper: `LLC2160_land_ocean_mask/utils.py` (compute_land_ocean_mask3d,
    apply_land_ocean_mask).
- Diagnostics: `diagnostics_and_report()` in this file.
"""
try:
    mask_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'LLC2160_land_ocean_mask'))
    if mask_module_path not in sys.path:
        sys.path.insert(0, mask_module_path)
    from utils import compute_land_ocean_mask3d, apply_land_ocean_mask
    has_mask_utils = True
except Exception:
    has_mask_utils = False


def resample_to_shape(src, target_shape, order=1):
    src_shape = src.shape
    if len(src_shape) != len(target_shape):
        raise ValueError("src and target must have same ndim")
    factors = [t / s for s, t in zip(src_shape, target_shape)]
    return zoom(src, factors, order=order)


def resample_mask_to_shape(mask, target_shape):
    """Resample boolean/int mask to target_shape using nearest-neighbor and
    return a boolean array.
    """
    mask = np.array(mask)
    if mask.shape == target_shape:
        return mask.astype(bool)
    try:
        resized = resample_to_shape(mask.astype(np.float32), target_shape, order=0)
        return resized.astype(bool)
    except Exception:
        # fallback: use numpy resize (may distort semantics)
        try:
            resized = np.resize(mask, target_shape)
            return np.array(resized).astype(bool)
        except Exception:
            return np.array(mask).astype(bool)


def main():
    try:
        # Load dataset
        url = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc2160_theta/llc2160_theta.idx"
        global ds
        ds = ovp.LoadDataset(url)
        
        global total_timesteps, hours_per_day
        total_timesteps = 10366
        hours_per_day = 24
        
        # Get the full range for x and y from the dataset
        logic_box = ds.db.getLogicBox()
        global x_range, y_range, z_range
        x_range = [logic_box[0][0], logic_box[1][0]]
        y_range = [logic_box[0][1], logic_box[1][1]]
        z_range = [logic_box[0][2], logic_box[1][2]]

        # attempt a single-step diagnostic run (timestep 0)
        timestep = 0
        start_time = datetime.now()
        data1 = ds.db.read(
            time=timestep,
            x=x_range,
            y=y_range,
            z=z_range,  
            quality=-6  
        )

        # If mask utilities are available, precompute a mask3d for this timestep
        mask3d = None
        if has_mask_utils:
            try:
                mask3d = compute_land_ocean_mask3d(time=timestep, quality=-6, use_surface_only=True)
            except Exception:
                mask3d = None

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        print(json.dumps({"quality_-6_read_time_seconds": execution_time}))

        # calculate the execution time for reading data at a lower quality (example -12)
        start_time = datetime.now()
        data2 = ds.db.read(
            time=timestep,
            x=x_range,
            y=y_range,
            z=z_range,  
            quality=-12
        )

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        print(json.dumps({"quality_-12_read_time_seconds": execution_time}))

        # calculate rmse between data1 and data2
        # first make sure they have the same shape
        if data1.shape != data2.shape:
            # resample data2 to match data1 shape
            data2 = resample_to_shape(data2, data1.shape, order=1)

        # If mask utilities are available, compute RMSE only over ocean points
        if has_mask_utils:
            print("Applying land/ocean mask for quality -12 data")
            try:
                data1_masked = apply_land_ocean_mask(data1, mask3d, land_value=np.nan)
                data2_masked = apply_land_ocean_mask(data2, mask3d, land_value=np.nan)
                
                diff = (data1_masked - data2_masked)
                rmse = float(np.sqrt(np.mean(diff ** 2)))
            except Exception:
                diff = (data1 - data2)
                rmse = float(np.sqrt(np.mean(diff ** 2)))
        else:
            diff = (data1 - data2)
            rmse = float(np.sqrt(np.mean(diff ** 2)))

        print(json.dumps({"rmse": rmse}))

    except Exception as e:
        print(json.dumps({"error": str(e)}))


def diagnostics_and_report(a, b, mask3d=None, save_map_path=None, thresholds=(0.1, 0.5, 1.0)):
    """Compute diagnostics between arrays a and b.

    Returns dict: rmse, bias, max_abs, median_abs, valid_count, and percent within thresholds.
    Optionally saves an absolute-error 2D image to save_map_path.
    """
    a = np.array(a)
    b = np.array(b)
    if a.shape != b.shape:
        raise ValueError(f"shapes must match for diagnostics (got {a.shape} vs {b.shape})")

    if mask3d is not None:
        try:
            # mask_bool == True  --> ocean
            # Use mask_bool directly so True selects ocean points.
            mask_bool = np.array(mask3d).astype(bool)
            valid = mask_bool
        except Exception:
            valid = ~np.isnan(a) & ~np.isnan(b)
    else:
        valid = ~np.isnan(a) & ~np.isnan(b)

    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        stats = {"rmse": None, "bias": None, "max_abs": None, "median_abs": None, "valid_count": 0}
        if save_map_path:
            try:
                import matplotlib.pyplot as plt
                plt.imsave(save_map_path, np.zeros((2, 2)), cmap='viridis')
            except Exception:
                pass
        return stats

    diff = (a - b).astype(float)
    diff_valid = diff[valid]

    rmse = float(np.sqrt(np.mean(diff_valid ** 2)))
    bias = float(np.mean(diff_valid))
    absdiff = np.abs(diff_valid)
    max_abs = float(np.max(absdiff))
    median_abs = float(np.median(absdiff))

    percent_within = {}
    for t in thresholds:
        pct = 100.0 * float(np.count_nonzero(absdiff <= t)) / len(absdiff)
        percent_within[f"pct_within_{t}"] = round(pct, 3)

    # save an error map (2D slice) if requested
    if save_map_path:
        try:
            import matplotlib.pyplot as plt
            if diff.ndim == 3:
                slice_idx = diff.shape[-1] // 2
                img = np.abs(diff[..., slice_idx])
            elif diff.ndim == 2:
                img = np.abs(diff)
            else:
                img = np.abs(diff.reshape((diff.shape[0], -1)))
            plt.imsave(save_map_path, img, cmap='viridis')
        except Exception:
            pass

    out = {
        "rmse": round(rmse, 6),
        "bias": round(bias, 6),
        "max_abs": round(max_abs, 6),
        "median_abs": round(median_abs, 6),
        "valid_count": valid_count,
    }
    # average absolute error. If a mask/valid selection was computed above,
    # compute the average over the compared (ocean) points only; otherwise
    # fall back to the full grid average.
    try:
        if 'valid' in locals() and valid is not None:
            valid_idx = np.nonzero(valid)
            if valid_idx[0].size:
                avg_abs_error = float(np.sum(np.abs((a - b)[valid_idx]))) / int(np.count_nonzero(valid))
            else:
                avg_abs_error = None
        else:
            total_points = int(np.prod(a.shape))
            avg_abs_error = float(np.sum(np.abs(a - b))) / total_points
    except Exception:
        avg_abs_error = None

    # percentage average error relative to ref range (use the same masking logic)
    try:
        if 'valid' in locals() and valid is not None and int(np.count_nonzero(valid)) > 0:
            ref_vals = a[valid]
        else:
            ref_vals = a
        ref_range = float(np.nanmax(ref_vals) - np.nanmin(ref_vals))
        pct_avg_error = (avg_abs_error / ref_range) * 100.0 if (avg_abs_error is not None and ref_range != 0) else None
    except Exception:
        pct_avg_error = None

    out['avg_abs_error'] = round(avg_abs_error, 8) if avg_abs_error is not None else None
    out['pct_avg_error'] = round(pct_avg_error, 6) if pct_avg_error is not None else None
    out.update(percent_within)
    return out


def run_sweep(output_csv_path=None, qualities=None, timesteps=None, save_maps=False, ref_quality=-6,
              x_range_override=None, y_range_override=None, z_range_override=None):
    """Run small sweep across qualities and timesteps and write CSV of diagnostics.

    Returns list of result dicts.
    """
    if qualities is None:
        qualities = [0, -3, -6, -4, -6, -8, -10, -12]
    if timesteps is None:
        timesteps = [0]
    # normalize timesteps: allow a triple [start,end,interval]
    range_spec = None
    if isinstance(timesteps, (list, tuple)) and len(timesteps) == 3 and all(isinstance(x, int) for x in timesteps):
        start_t, end_t, step_t = timesteps
        if step_t == 0:
            timesteps_list = [start_t]
            interval_t = 0
        else:
            step_t = max(1, step_t)
            if end_t >= start_t:
                timesteps_list = list(range(start_t, end_t + 1, step_t))
            else:
                timesteps_list = list(range(start_t, end_t - 1, -step_t))
            interval_t = step_t
        range_spec = (start_t, end_t, interval_t)
    else:
        timesteps_list = list(timesteps)

    base_dir = os.path.join(os.path.dirname(__file__), '..','..','..','ai_data')
    os.makedirs(base_dir, exist_ok=True)
    maps_dir = os.path.join(base_dir, 'accuracy_error_maps')
    if save_maps:
        os.makedirs(maps_dir, exist_ok=True)

    if output_csv_path is None:
        output_csv_path = os.path.join(base_dir, 'accuracy_sweep.csv')

    # ensure dataset loaded and geometry available
    url = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc2160_theta/llc2160_theta.idx"
    global ds, x_range, y_range, z_range
    if 'ds' not in globals() or ds is None:
        if ovp is None:
            raise RuntimeError('OpenVisus binding not available')
        ds = ovp.LoadDataset(url)
        logic_box = ds.db.getLogicBox()
        # compute local default subset ranges if overrides are not provided.
        # The user requested a small middle subset for x/y by default. We use
        # simple fractional slices of the dataset extents here; these are
        # integer indices.
        lb0 = logic_box[0]
        lb1 = logic_box[1]
        if x_range_override is None:
            # default to full x extent
            x_range_local = [int(lb0[0]), int(lb1[0])]
        else:
            x_range_local = list(x_range_override)

        if y_range_override is None:
            # default to full y extent
            y_range_local = [int(lb0[1]), int(lb1[1])]
        else:
            y_range_local = list(y_range_override)

        if z_range_override is None:
            z_range_local = [int(lb0[2]), int(lb1[2])]
        else:
            z_range_local = list(z_range_override)

    # attempt mask functions
    mask_available = False
    try:
        mask_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'LLC2160_land_ocean_mask'))
        if mask_module_path not in sys.path:
            sys.path.insert(0, mask_module_path)
        from utils import compute_land_ocean_mask3d, apply_land_ocean_mask  # type: ignore
        mask_available = True
    except Exception:
        mask_available = False

    results = []

    # Prepare CSV writer for incremental updates so results are visible as
    # each quality finishes.
    import csv
    # build header columns (include pct_within keys for default thresholds)
    thresholds = (0.1, 0.5, 1.0)
    pct_keys = [f"pct_within_{t}" for t in thresholds]
    # include spatial subset columns so CSV rows show which x/y/z were used
    # We'll write a stable, human-friendly CSV. The CSV will be opened in append
    # mode so subsequent runs add rows rather than overwrite the file. We still
    # compute and print pct_within and valid_count for diagnostics, but do not
    # include them as CSV columns per user request.
    csv_keys = ['quality/resolution', 'start_timestep', 'end_timestep', 'time_interval', 'x_range', 'y_range', 'z_range', 'total_read_time_seconds', 'absolute_data_points', 'min_temperature_ocean', 'max_temperature_ocean', 'rmse', 'average_signed_error(bias)', 'median_absolute_error', 'average_absolute_error', 'percentage_average_error']
    # open in append mode; create file and header if missing
    file_exists = os.path.exists(output_csv_path)
    csv_file = open(output_csv_path, 'a', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_keys)
    if not file_exists:
        csv_writer.writeheader()
        csv_file.flush()

    # Ensure local ranges exist if ds was already initialized earlier
    if 'x_range_local' not in locals():
        if x_range_override is not None:
            x_range_local = list(x_range_override)
        elif 'x_range' in globals():
            x_range_local = list(x_range)
        else:
            logic_box = ds.db.getLogicBox()
            lb0 = logic_box[0]
            lb1 = logic_box[1]
            x_range_local = [int(lb0[0]), int(lb1[0])]

    if 'y_range_local' not in locals():
        if y_range_override is not None:
            y_range_local = list(y_range_override)
        elif 'y_range' in globals():
            y_range_local = list(y_range)
        else:
            logic_box = ds.db.getLogicBox()
            lb0 = logic_box[0]
            lb1 = logic_box[1]
            y_range_local = [int(lb0[1]), int(lb1[1])]

    if 'z_range_local' not in locals():
        if z_range_override is not None:
            z_range_local = list(z_range_override)
        elif 'z_range' in globals():
            z_range_local = list(z_range)
        else:
            logic_box = ds.db.getLogicBox()
            lb0 = logic_box[0]
            lb1 = logic_box[1]
            z_range_local = [int(lb0[2]), int(lb1[2])]

    # If a range triple was provided, run aggregated mode: perform reads for
    # every timestep in timesteps_list but produce only one CSV row per
    # quality summarizing the total read time across the range. This avoids
    # writing one row per timestep when the caller asked for a range.
    if range_spec is not None:
        start_ts, end_ts, interval_ts = range_spec
        # initialize aggregates for each quality
        aggregates = {q: {'read_times': [], 'pre_points': 0, 'rmse_list': [], 'avg_abs_list': []} for q in qualities}

        for t in timesteps_list:
            # read reference and prepare mask as in the single-step flow
            try:
                ref = ds.db.read(time=t, x=x_range_local, y=y_range_local, z=z_range_local, quality=ref_quality)
            except Exception as e:
                print(f"Failed to read reference for t={t} in aggregated range: {e}")
                # mark error for each quality and continue
                for q in qualities:
                    aggregates[q]['read_times'].append(None)
                continue

            # prepare mask_for_ref for this timestep
            mask_for_ref = None
            if mask_available:
                try:
                    try:
                        raw_mask = compute_land_ocean_mask3d(time=t, quality=ref_quality, use_surface_only=True,
                                                             x_range=x_range_local, y_range=y_range_local, z_range=z_range_local)
                    except TypeError:
                        raw_mask = compute_land_ocean_mask3d(time=t, quality=ref_quality, use_surface_only=True)
                        try:
                            lb0, lb1 = ds.db.getLogicBox()
                            x0_off = int(x_range_local[0] - lb0[0])
                            x1_off = int(x_range_local[1] - lb0[0])
                            y0_off = int(y_range_local[0] - lb0[1])
                            y1_off = int(y_range_local[1] - lb0[1])
                            z0_off = int(z_range_local[0] - lb0[2])
                            z1_off = int(z_range_local[1] - lb0[2])
                            z0_off = max(0, z0_off); z1_off = max(z0_off, min(raw_mask.shape[0], z1_off))
                            y0_off = max(0, y0_off); y1_off = max(y0_off, min(raw_mask.shape[1], y1_off))
                            x0_off = max(0, x0_off); x1_off = max(x0_off, min(raw_mask.shape[2], x1_off))
                            raw_mask = raw_mask[z0_off:z1_off, y0_off:y1_off, x0_off:x1_off]
                        except Exception:
                            pass

                    if raw_mask is None:
                        raise RuntimeError('compute_land_ocean_mask3d returned None')
                    raw_mask = np.asarray(raw_mask)
                    if raw_mask.shape == ref.shape:
                        mask_for_ref = raw_mask.astype(bool)
                    else:
                        if raw_mask.ndim == 2 and raw_mask.shape == (ref.shape[1], ref.shape[2]):
                            mask_for_ref = np.broadcast_to(raw_mask[np.newaxis, ...], ref.shape).astype(bool)
                        elif raw_mask.ndim == 3 and raw_mask.shape[0] == 1 and raw_mask.shape[1:] == (ref.shape[1], ref.shape[2]):
                            mask_for_ref = np.broadcast_to(raw_mask, ref.shape).astype(bool)
                        else:
                            raise RuntimeError('compute_land_ocean_mask3d returned incompatible mask shape for aggregated range')
                except Exception:
                    raise

            # for each quality read and accumulate times/diagnostics
            for q in qualities:
                try:
                    t0 = perf_counter()
                    test = ds.db.read(time=t, x=x_range_local, y=y_range_local, z=z_range_local, quality=q)
                    dt = perf_counter() - t0
                except Exception as e:
                    print(f"Read error q={q} t={t} in aggregated range: {e}")
                    aggregates[q]['read_times'].append(None)
                    continue

                try:
                    pre_points = int(np.prod(test.shape))
                except Exception:
                    pre_points = 0
                aggregates[q]['pre_points'] += pre_points
                aggregates[q]['read_times'].append(dt)

                # compute diagnostics per-step and accumulate rmse/avg_abs
                try:
                    if ref.shape != test.shape:
                        try:
                            test_r = resample_to_shape(test, ref.shape, order=1)
                        except Exception:
                            test_r = np.resize(test, ref.shape)
                    else:
                        test_r = test
                    diag = diagnostics_and_report(ref, test_r, mask3d=mask_for_ref)
                    if diag.get('rmse') is not None:
                        aggregates[q]['rmse_list'].append(diag.get('rmse'))
                    if diag.get('avg_abs_error') is not None:
                        aggregates[q]['avg_abs_list'].append(diag.get('avg_abs_error'))
                except Exception:
                    pass

        # after iterating timesteps_list, write one summary row per quality
        for q in qualities:
            read_times = [r for r in aggregates[q]['read_times'] if r is not None]
            if read_times:
                total_read = round(float(sum(read_times)), 6)
                mean_read = float(np.mean(read_times))
                med_read = float(np.median(read_times))
            else:
                total_read = None
                mean_read = None
                med_read = None

            mean_rmse = None
            if aggregates[q]['rmse_list']:
                mean_rmse = float(np.mean(aggregates[q]['rmse_list']))

            csv_row = {
                'quality/resolution': q,
                'start_timestep': start_ts,
                'end_timestep': end_ts,
                'time_interval': interval_ts,
                'x_range': str(x_range_local),
                'y_range': str(y_range_local),
                'z_range': str(z_range_local),
                # store the total time across the range here
                'total_read_time_seconds': total_read,
                'absolute_data_points': aggregates[q]['pre_points'],
                'min_temperature_ocean': None,
                'max_temperature_ocean': None,
                'rmse': (fmt_temp(mean_rmse) if mean_rmse is not None else None),
                'average_signed_error(bias)': None,
                'median_absolute_error': None,
                'average_absolute_error': (round(float(np.mean(aggregates[q]['avg_abs_list'])), 8) if aggregates[q]['avg_abs_list'] else None),
                'percentage_average_error': None
            }
            try:
                csv_writer.writerow({k: csv_row.get(k, None) for k in csv_keys})
                csv_file.flush()
            except Exception:
                pass
            results.append(csv_row)

        # finished aggregated mode; return results to avoid per-timestep rows
        try:
            csv_file.close()
        except Exception:
            pass
        print(f"Wrote aggregated sweep summary for timesteps {start_ts}:{end_ts}:{interval_ts} to {output_csv_path}")
        return results

    # Normal (per-timestep) sweep mode: iterate each requested timestep and
    # perform reference + quality comparisons. This block is only executed
    # when a range triple was not provided (range_spec is None).
    if range_spec is None:
        for t in timesteps_list:
            # Compute/read the reference once per timestep so we can compute a
            # reusable mask resampled to the reference grid. This avoids
            # recomputing the (potentially expensive) land/ocean mask for every
            # quality.
            try:
                ref_start = perf_counter()
                ref = ds.db.read(time=t, x=x_range_local, y=y_range_local, z=z_range_local, quality=ref_quality)
                ref_end = perf_counter()
                ref_read_time = round(ref_end - ref_start, 4)
            except Exception as e:
                print(f"Failed to read reference for t={t}, q={ref_quality}: {e}")
                # still iterate qualities to report errors
                for q in qualities:
                    if range_spec is not None:
                        start_ts, end_ts, interval_ts = range_spec
                    else:
                        start_ts, end_ts, interval_ts = (t, t, 0)

                    row = {
                        'quality/resolution': q,
                        'start_timestep': start_ts,
                        'end_timestep': end_ts,
                        'time_interval': interval_ts,
                        'x_range': str(x_range_local),
                        'y_range': str(y_range_local),
                        'z_range': str(z_range_local),
                        'error': str(e)
                    }
                    results.append(row)
                    # write incremental error row
                    try:
                        csv_writer.writerow({k: row.get(k, None) for k in csv_keys})
                        csv_file.flush()
                    except Exception:
                        pass
                continue

            # Precompute mask_for_ref once per timestep and enforce strict shape
            # semantics: the helper MUST return a mask that either matches the
            # reference grid shape exactly (nz, ny, nx) or is a surface-only mask
            # with shape (1, ny, nx) or (ny, nx) which will be broadcast to depth.
            # We intentionally avoid silent resampling or numpy.resize fallbacks
            # because those can flip coastline cells and produce incorrect
            # ocean/land selections. If the helper cannot return an aligned mask,
            # raise an explicit error with guidance.
            mask_for_ref = None
            if mask_available:
                # First, try to ask the helper for a mask on the requested subset
                # at the reference quality. If the helper does not accept ranges,
                # fall back to requesting the full-domain mask and cropping it.
                try:
                    try:
                        raw_mask = compute_land_ocean_mask3d(time=t, quality=ref_quality, use_surface_only=True,
                                                             x_range=x_range_local, y_range=y_range_local, z_range=z_range_local)
                    except TypeError:
                        # helper does not accept ranges: request full mask then crop
                        raw_mask = compute_land_ocean_mask3d(time=t, quality=ref_quality, use_surface_only=True)
                        try:
                            lb0, lb1 = ds.db.getLogicBox()
                            x0_off = int(x_range_local[0] - lb0[0])
                            x1_off = int(x_range_local[1] - lb0[0])
                            y0_off = int(y_range_local[0] - lb0[1])
                            y1_off = int(y_range_local[1] - lb0[1])
                            z0_off = int(z_range_local[0] - lb0[2])
                            z1_off = int(z_range_local[1] - lb0[2])
                            # clamp indices to mask shape
                            z0_off = max(0, z0_off); z1_off = max(z0_off, min(raw_mask.shape[0], z1_off))
                            y0_off = max(0, y0_off); y1_off = max(y0_off, min(raw_mask.shape[1], y1_off))
                            x0_off = max(0, x0_off); x1_off = max(x0_off, min(raw_mask.shape[2], x1_off))
                            raw_mask = raw_mask[z0_off:z1_off, y0_off:y1_off, x0_off:x1_off]
                        except Exception:
                            # cropping failed; we will still validate shape below and
                            # raise a clear error if incompatible.
                            pass

                    if raw_mask is None:
                        raise RuntimeError('compute_land_ocean_mask3d returned None')

                    raw_mask = np.asarray(raw_mask)

                    # Accept exact match
                    if raw_mask.shape == ref.shape:
                        mask_for_ref = raw_mask.astype(bool)
                    else:
                        # Accept surface-only masks: (ny, nx) or (1, ny, nx)
                        if raw_mask.ndim == 2 and raw_mask.shape == (ref.shape[1], ref.shape[2]):
                            mask_for_ref = np.broadcast_to(raw_mask[np.newaxis, ...], ref.shape).astype(bool)
                        elif raw_mask.ndim == 3 and raw_mask.shape[0] == 1 and raw_mask.shape[1:] == (ref.shape[1], ref.shape[2]):
                            mask_for_ref = np.broadcast_to(raw_mask, ref.shape).astype(bool)
                        else:
                            raise RuntimeError(
                                f"compute_land_ocean_mask3d returned mask with shape {raw_mask.shape}, which is incompatible with reference shape {ref.shape}.\n"
                                "Allowed shapes: exactly ref.shape, or a surface-only mask with shape (1, ny, nx) or (ny, nx) that will be broadcast.\n"
                                "Call compute_land_ocean_mask3d(time=..., quality=ref_quality, x_range=..., y_range=..., z_range=...) to obtain an aligned mask."
                            )
                except Exception:
                    # Bubble up the error to the caller so the user/developer sees
                    # why the sweep cannot proceed with an unreliable mask.
                    raise

            # Write an explicit reference row (self-compare) so the CSV contains
            # an entry for the reference quality itself. Use diagnostics between
            # ref and ref to obtain valid_count and zeroed metrics.
            try:
                diag_ref = diagnostics_and_report(ref, ref, mask3d=mask_for_ref)
                # Build a CSV-friendly row. We include the total read time for the
                # reference (same as ref_read_time), number of data points in the
                # coarser sample before resampling, and the sample min/max values.
                pre_points = None
                sample_min = None
                sample_max = None
                try:
                    # attempt to capture pre-resample stats from the raw ref read
                    pre_points = int(np.prod(ref.shape))
                    sample_min = float(np.nanmin(ref))
                    sample_max = float(np.nanmax(ref))
                except Exception:
                    pass

                # Format numeric fields with units where appropriate for CSV
                def fmt_temp(v, ndigits=6):
                    return None if v is None else f"{round(v, ndigits)}°C"

                def fmt_pct(v, ndigits=6):
                    return None if v is None else f"{round(v, ndigits)}%"

                if range_spec is not None:
                    start_ts, end_ts, interval_ts = range_spec
                else:
                    start_ts, end_ts, interval_ts = (t, t, 0)

                ref_row = {
                    'quality/resolution': ref_quality,
                    'start_timestep': start_ts,
                    'end_timestep': end_ts,
                    'time_interval': interval_ts,
                    'x_range': str(x_range_local),
                    'y_range': str(y_range_local),
                    'z_range': str(z_range_local),
                    'total_read_time_seconds': ref_read_time,
                    'absolute_data_points': pre_points,
                    # For the reference row, ocean extrema are computed when
                    # mask_for_ref exists. We do not write raw minimum/maximum
                    # values to the CSV per user request.
                    'min_temperature_ocean': None,
                    'max_temperature_ocean': None,
                    'rmse': fmt_temp(diag_ref.get('rmse')),
                    'average_signed_error(bias)': fmt_temp(diag_ref.get('bias')),
                    'median_absolute_error': fmt_temp(diag_ref.get('median_abs')),
                    'average_absolute_error': fmt_temp(diag_ref.get('avg_abs_error')),
                    'percentage_average_error': fmt_pct(diag_ref.get('pct_avg_error'))
                }
                results.append(ref_row)
                try:
                    # If a mask is available, compute ocean-only extrema for the
                    # reference sample and write formatted °C values.
                    if mask_for_ref is not None:
                        try:
                            mask_bool = np.array(mask_for_ref).astype(bool)
                            # mask_bool True == ocean
                            ocean_vals = ref[mask_bool]
                            if ocean_vals.size:
                                ref_row['min_temperature_ocean'] = fmt_temp(float(np.nanmin(ocean_vals)))
                                ref_row['max_temperature_ocean'] = fmt_temp(float(np.nanmax(ocean_vals)))
                        except Exception:
                            pass

                    csv_writer.writerow({k: ref_row.get(k, None) for k in csv_keys})
                    csv_file.flush()
                except Exception:
                    pass

            except Exception:
                pass

        for q in qualities:
            try:
                start = perf_counter()
                # We already read the reference above and have mask_for_ref.
                test_read_start = perf_counter()
                test = ds.db.read(time=t, x=x_range_local, y=y_range_local, z=z_range_local, quality=q)
                test_read_end = perf_counter()

                pre_points = None
                sample_min = None
                sample_max = None
                try:
                    pre_points = int(np.prod(test.shape))
                    sample_min = float(np.nanmin(test))
                    sample_max = float(np.nanmax(test))
                except Exception:
                    pass

                if ref.shape != test.shape:
                    try:
                        test = resample_to_shape(test, ref.shape, order=1)
                    except Exception:
                        test = np.resize(test, ref.shape)

                # Reuse the precomputed/resampled mask_for_ref for diagnostics.
                mask_local = mask_for_ref

                map_path = None
                if save_maps:
                    map_path = os.path.join(maps_dir, f'err_q{q}_t{t}.png')

                diag = diagnostics_and_report(ref, test, mask3d=mask_local, save_map_path=map_path)

                # total read time for this test (how long the ds.db.read call took)
                total_read_time = round(test_read_end - test_read_start, 4)

                # Build the printed result (includes pct_within and valid_count)
                # range fields for this row
                if range_spec is not None:
                    start_ts, end_ts, interval_ts = range_spec
                else:
                    start_ts, end_ts, interval_ts = (t, t, 0)

                result = {
                    'quality': q,
                    'start_timestep': start_ts,
                    'end_timestep': end_ts,
                    'time_interval': interval_ts,
                    'x_range': str(x_range_local),
                    'y_range': str(y_range_local),
                    'z_range': str(z_range_local),
                    'total_read_time_seconds': total_read_time,
                    'pre_resample_points': pre_points,
                    'sample_min': sample_min,
                    'sample_max': sample_max,
                    'rmse': diag.get('rmse'),
                    'bias': diag.get('bias'),
                    'max_abs': diag.get('max_abs'),
                    'median_abs': diag.get('median_abs'),
                    'avg_abs_error': diag.get('avg_abs_error'),
                    'percentage_average_error': diag.get('pct_avg_error')
                }
                # include percent-within keys
                # Keep printing pct_within and valid_count for diagnostics, but do
                # not add them to the CSV (per user request).
                for k, v in diag.items():
                    if k.startswith('pct_within'):
                        result[k] = v
                result['valid_count'] = diag.get('valid_count')

                results.append(result)
                print(json.dumps(result))
                # Write CSV-friendly row mapping printed result values to the
                # user-friendly CSV column names and formatting units.
                csv_row = {
                    'quality/resolution': q,
                    'start_timestep': start_ts,
                    'end_timestep': end_ts,
                    'time_interval': interval_ts,
                    'x_range': str(x_range_local),
                    'y_range': str(y_range_local),
                    'z_range': str(z_range_local),
                    'total_read_time_seconds': total_read_time,
                    'absolute_data_points': pre_points,
                    # compute ocean-only extrema using the precomputed mask
                    'min_temperature_ocean': None,
                    'max_temperature_ocean': None,
                    'rmse': fmt_temp(diag.get('rmse')),
                    'average_signed_error(bias)': fmt_temp(diag.get('bias')),
                    'median_absolute_error': fmt_temp(diag.get('median_abs')),
                    'average_absolute_error': fmt_temp(diag.get('avg_abs_error')),
                    'percentage_average_error': fmt_pct(diag.get('pct_avg_error'))
                }
                # populate ocean extrema if mask is available
                if mask_local is not None:
                    try:
                        mask_bool = np.array(mask_local).astype(bool)
                        # mask_bool True == ocean
                        ocean_vals = test[mask_bool]
                        if ocean_vals.size:
                            csv_row['min_temperature_ocean'] = fmt_temp(float(np.nanmin(ocean_vals)))
                            csv_row['max_temperature_ocean'] = fmt_temp(float(np.nanmax(ocean_vals)))
                    except Exception:
                        pass

                try:
                    csv_writer.writerow({k: csv_row.get(k, None) for k in csv_keys})
                    csv_file.flush()
                except Exception:
                    pass

            except Exception as e:
                print(f"Sweep error q={q} t={t}: {e}")
                results.append({'quality': q, 'timestep': t, 'error': str(e)})

    # close incremental CSV file
    try:
        csv_file.close()
        print(f"Wrote sweep CSV to {output_csv_path}")
    except Exception:
        pass

    return results


if __name__ == '__main__':
    # Run the diagnostic main function only. To run the full sweep, import
    # this module and call run_sweep(...) with explicit parameters to
    # avoid accidental heavy/long-running operations when importing.

    main()

