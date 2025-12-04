import numpy as np
import matplotlib.pyplot as plt
import os
import json

# Paths
cache_path = "/Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app/ai_data/data_cache/dyamond_llc2160/data_20251202_190204.npz"
plots_dir = "/Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app/ai_data/plots/dyamond_llc2160"
os.makedirs(plots_dir, exist_ok=True)

# Load cached data
data = np.load(cache_path, allow_pickle=True)
print("Available keys:", data.files)

# Extract metadata and arrays
variable_name = str(data["variable_name"])
variable_unit = str(data["variable_unit"])
quality = int(data["quality_level"])
lat_min, lat_max = data["lat_range"]
lon_min, lon_max = data["lon_range"]
date_may = str(data["date_may"])
date_nov = str(data["date_nov"])

temp_may = data["temp_may"]
temp_nov = data["temp_nov"]
anom = data["anomaly"]

# Stats and percentiles (min, max, mean, std)
stats_may = data["stats_may"]  # [min, max, mean, std]
stats_nov = data["stats_nov"]
stats_anom = data["stats_anom"]

# Percentiles arrays in order [p5, p25, p50, p75, p95]
percs_may = data["percentiles_may"]
percs_nov = data["percentiles_nov"]
percs_anom = data["percentiles_anom"]

# Histograms (precomputed)
edges_val = data["hist_edges_value"]
counts_may = data["hist_counts_may"]
counts_nov = data["hist_counts_nov"]
edges_anom = data["hist_edges_anom"]
counts_anom = data["hist_counts_anom"]

# Prepare display settings
finite_may = temp_may[np.isfinite(temp_may)]
finite_nov = temp_nov[np.isfinite(temp_nov)]
finite_anom = anom[np.isfinite(anom)]

# Use a fixed, comparable color scale for value maps (request: 0–35 °C)
vmin_vals, vmax_vals = 0.0, 35.0

# Symmetric color scale for anomaly (zero-centered) using ±max(|min|,|max|)
if finite_anom.size:
    max_abs_anom = max(abs(np.nanmin(finite_anom)), abs(np.nanmax(finite_anom)))
    if not np.isfinite(max_abs_anom) or max_abs_anom == 0:
        max_abs_anom = 1.0
else:
    max_abs_anom = 1.0

# Configure colormaps
cmap_val = plt.get_cmap("viridis").copy()
cmap_val.set_bad(color="lightgray")
cmap_anom = plt.get_cmap("coolwarm").copy()
cmap_anom.set_bad(color="lightgray")

# Helper to format stats text
def fmt_stats_row(label, stats_array):
    mn, mx, mean, std = stats_array
    return f"{label:8s}  min={mn:6.2f}  max={mx:6.2f}  mean={mean:6.2f}  std={std:6.2f}"

# Helper to add simple lon/lat gridlines
def add_lonlat_grid(ax, xmin, xmax, ymin, ymax):
    def pick_step(span):
        if span > 60: return 20
        if span > 30: return 10
        if span > 10: return 5
        if span > 5: return 2
        return 1
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    lon_step = pick_step(xmax - xmin)
    lat_step = pick_step(ymax - ymin)
    # Ensure ticks cover bounds
    lon_ticks = np.arange(np.ceil(xmin / lon_step) * lon_step, np.floor(xmax / lon_step) * lon_step + 0.5 * lon_step, lon_step)
    lat_ticks = np.arange(np.ceil(ymin / lat_step) * lat_step, np.floor(ymax / lat_step) * lat_step + 0.5 * lat_step, lat_step)
    if lon_ticks.size >= 2:
        ax.set_xticks(lon_ticks)
    if lat_ticks.size >= 2:
        ax.set_yticks(lat_ticks)
    ax.grid(True, which="both", color="k", alpha=0.15, linewidth=0.5)

# Plot 1: May temperature map (surface, z=0)
plt.figure(figsize=(10, 6))
# Using z=0 (surface level), no resolution reduction applied
im = plt.imshow(temp_may, origin="lower",
                extent=[lon_min, lon_max, lat_min, lat_max],
                cmap=cmap_val, vmin=vmin_vals, vmax=vmax_vals, aspect="auto")
cb = plt.colorbar(im, pad=0.02)
cb.set_label(f"{variable_name} ({variable_unit})")
plt.title(f"Near-surface temperature (z=0)\nIndian Ocean — {date_may}")
plt.xlabel("Longitude (°E)")
plt.ylabel("Latitude (°N)")
# Add subtle stats as text box
txt = fmt_stats_row("May", stats_may)
ax = plt.gca()
ax.text(0.01, 0.02, txt, transform=ax.transAxes,
        fontsize=9, va="bottom", ha="left",
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
add_lonlat_grid(ax, lon_min, lon_max, lat_min, lat_max)
out1 = os.path.join(plots_dir, "plot_1_20251202_190204_revised.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()

# Plot 2: Nov temperature map (surface, z=0)
plt.figure(figsize=(10, 6))
# Using z=0 (surface level), no resolution reduction applied
im = plt.imshow(temp_nov, origin="lower",
                extent=[lon_min, lon_max, lat_min, lat_max],
                cmap=cmap_val, vmin=vmin_vals, vmax=vmax_vals, aspect="auto")
cb = plt.colorbar(im, pad=0.02)
cb.set_label(f"{variable_name} ({variable_unit})")
plt.title(f"Near-surface temperature (z=0)\nIndian Ocean — {date_nov}")
plt.xlabel("Longitude (°E)")
plt.ylabel("Latitude (°N)")
txt = fmt_stats_row("Nov", stats_nov)
ax = plt.gca()
ax.text(0.01, 0.02, txt, transform=ax.transAxes,
        fontsize=9, va="bottom", ha="left",
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
add_lonlat_grid(ax, lon_min, lon_max, lat_min, lat_max)
out2 = os.path.join(plots_dir, "plot_2_20251202_190204_revised.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()

# Plot 3: Anomaly map (Nov - May)
plt.figure(figsize=(10, 6))
# Using z=0 (surface level), no resolution reduction applied
im = plt.imshow(anom, origin="lower",
                extent=[lon_min, lon_max, lat_min, lat_max],
                cmap=cmap_anom, vmin=-max_abs_anom, vmax=max_abs_anom, aspect="auto")
cb = plt.colorbar(im, pad=0.02)
cb.set_label("Temperature anomaly (°C)\n(Nov − May)")
plt.title(f"Near-surface temperature anomaly (z=0)\nIndian Ocean — {date_nov} minus {date_may}")
plt.xlabel("Longitude (°E)")
plt.ylabel("Latitude (°N)")
txt = fmt_stats_row("Anom", stats_anom)
ax = plt.gca()
ax.text(0.01, 0.02, txt, transform=ax.transAxes,
        fontsize=9, va="bottom", ha="left",
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
add_lonlat_grid(ax, lon_min, lon_max, lat_min, lat_max)
out3 = os.path.join(plots_dir, "plot_3_20251202_190204_revised.png")
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()

# Plot 4: Histograms with percentile markers (May & Nov combined)
plt.figure(figsize=(10, 6))
# Value distributions (May & Nov)
# Convert edges to centers for step-like plotting
centers_val = 0.5 * (edges_val[:-1] + edges_val[1:])
plt.plot(centers_val, counts_may, drawstyle="steps-mid", label=f"May {date_may}", color="#1f77b4", linewidth=1.5)
plt.plot(centers_val, counts_nov, drawstyle="steps-mid", label=f"Nov {date_nov}", color="#ff7f0e", linewidth=1.5)

# Percentile markers
def add_percentile_lines(values, label_prefix, color):
    p5, p25, p50, p75, p95 = values
    # lines for percentiles
    for v, ls in zip([p5, p25, p50, p75, p95], ["--", ":", "-", ":", "--"]):
        plt.axvline(v, color=color, linestyle=ls, alpha=0.6, linewidth=1.0)
    # mean line (dash-dot)
    plt.axvline(stats_may[2] if label_prefix.lower().startswith("may") else stats_nov[2],
                color=color, linestyle="--", alpha=0.8, linewidth=1.0, dashes=(4, 2))
    # Annotate median
    ymax = plt.ylim()[1]
    plt.text(p50, ymax * 0.95, f"{label_prefix} median={p50:.2f}°C",
             color=color, rotation=90, va="top", ha="right", fontsize=9, alpha=0.8)

add_percentile_lines(percs_may, "May", "#1f77b4")
add_percentile_lines(percs_nov, "Nov", "#ff7f0e")

plt.xlabel(f"{variable_name} ({variable_unit})")
plt.ylabel("Pixel count")
plt.title("Near-surface temperature distribution over Indian Ocean (masked to ocean)\nPercentiles: 5, 25, 50, 75, 95; dashed line shows mean")
plt.legend()
out4 = os.path.join(plots_dir, "plot_4_20251202_190204_revised.png")
plt.savefig(out4, dpi=150, bbox_inches="tight")
plt.close()

# Plot 5: Domain stats table (min, max, mean, std) for May, Nov, Anomaly
plt.figure(figsize=(8, 3.8))
plt.axis("off")
rows = [
    ["Field", "min (°C)", "max (°C)", "mean (°C)", "std (°C)"],
    ["May (z=0)", f"{stats_may[0]:.2f}", f"{stats_may[1]:.2f}", f"{stats_may[2]:.2f}", f"{stats_may[3]:.2f}"],
    ["Nov (z=0)", f"{stats_nov[0]:.2f}", f"{stats_nov[1]:.2f}", f"{stats_nov[2]:.2f}", f"{stats_nov[3]:.2f}"],
    ["Anomaly (Nov−May)", f"{stats_anom[0]:.2f}", f"{stats_anom[1]:.2f}", f"{stats_anom[2]:.2f}", f"{stats_anom[3]:.2f}"],
]
table = plt.table(cellText=rows, cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
plt.title("Summary statistics across Indian Ocean domain (ocean-only; z=0)", pad=10)
out5 = os.path.join(plots_dir, "plot_5_20251202_190204_revised.png")
plt.savefig(out5, dpi=150, bbox_inches="tight")
plt.close()

# NEW Plot 6: Panel with three histograms (May, Nov, Anomaly) with consistent binning and percentile/mean lines
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

# Helper to draw a histogram from precomputed counts/edges
def draw_hist(ax, edges, counts, title, color, percs, mean_val, unit="°C"):
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    ax.bar(centers, counts, width=widths, color=color, alpha=0.6, edgecolor=color, linewidth=0.6)
    # Vertical lines for p5, p25, p50, p75, p95
    p5, p25, p50, p75, p95 = percs
    for v, ls in zip([p5, p25, p50, p75, p95], ["--", ":", "-", ":", "--"]):
        ax.axvline(v, color="k", linestyle=ls, alpha=0.8, linewidth=1.0)
    # Mean line
    ax.axvline(mean_val, color="k", linestyle="--", linewidth=1.0, dashes=(4, 2))
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(f"{variable_name} ({unit})")
    ax.set_ylabel("Pixel count")
    ax.set_xlim(edges[0], edges[-1])
    # Add a small legend proxy
    ax.text(0.98, 0.95, "p5, p25, p50, p75, p95\n— mean",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

# May (uses value edges)
draw_hist(axes[0], edges_val, counts_may,
          title=f"May {date_may}", color="#1f77b4",
          percs=percs_may, mean_val=float(stats_may[2]), unit=variable_unit)

# Nov (uses value edges; same bins as May for consistency)
draw_hist(axes[1], edges_val, counts_nov,
          title=f"Nov {date_nov}", color="#ff7f0e",
          percs=percs_nov, mean_val=float(stats_nov[2]), unit=variable_unit)

# Anomaly (own symmetric distribution)
draw_hist(axes[2], edges_anom, counts_anom,
          title="Anomaly (Nov − May)", color="#8c564b",
          percs=percs_anom, mean_val=float(stats_anom[2]), unit="°C")

# Caption with percentile values
def percs_str(name, percs, mean_val):
    p5, p25, p50, p75, p95 = percs
    return (f"{name}: p5={p5:.2f}, p25={p25:.2f}, median={p50:.2f}, "
            f"p75={p75:.2f}, p95={p95:.2f}, mean={mean_val:.2f}")

caption = (
    "Percentiles over Indian Ocean (ocean mask; z=0):\n"
    + percs_str("May", percs_may, stats_may[2]) + " °C; "
    + percs_str("Nov", percs_nov, stats_nov[2]) + " °C; "
    + percs_str("Anom", percs_anom, stats_anom[2]) + " °C"
)
fig.suptitle("Distributions: May, Nov, and Anomaly (consistent binning for May/Nov; zero-centered for Anomaly)", fontsize=12, y=1.02)
fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=9)

out6 = os.path.join(plots_dir, "plot_6_20251202_190204_revised.png")
fig.savefig(out6, dpi=150, bbox_inches="tight")
plt.close(fig)

print(json.dumps({
    "status": "success",
    "plots_created": [out1, out2, out3, out4, out5, out6],
    "notes": "Maps: fixed 0–35 °C scale for May/Nov; anomaly uses symmetric zero-centered limits. Added gridlines, removed Q from titles. New 3-panel distribution figure with percentile and mean markers and caption."
}))