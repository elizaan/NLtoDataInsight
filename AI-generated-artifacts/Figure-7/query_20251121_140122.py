import openvisuspy as ovp
import numpy as np
import json
import xarray as xr
from datetime import datetime

# Define the Agulhas Current region
lat_range = [-40, -30]
lon_range = [15, 35]

# Convert lat/lon to x/y indices
def latlon_to_xy(lat_range, lon_range):
    geo_file = "/Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app/src/datasets/llc2160_latlon.nc"
    ds = xr.open_dataset(geo_file)
    lat_center = ds["latitude"].values
    lon_center = ds["longitude"].values
    
    mask = (
        (lat_center >= lat_range[0]) & (lat_center <= lat_range[1]) &
        (lon_center >= lon_range[0]) & (lon_center <= lon_range[1])
    )
    
    y_indices, x_indices = np.where(mask)
    x_min, x_max = int(x_indices.min()), int(x_indices.max()) + 1
    y_min, y_max = int(y_indices.min()), int(y_indices.max()) + 1
    
    return [x_min, x_max], [y_min, y_max]

x_range, y_range = latlon_to_xy(lat_range, lon_range)

# Temporal range: 14 months from dataset start
dataset_start = datetime.strptime("2020-01-20", "%Y-%m-%d")
timestep_start = 0
timestep_end = 10366  # Full range for 14 months

# Load datasets
temperature_url = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc2160_theta/llc2160_theta.idx"
eastwest_velocity_url = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc2160_arco/visus.idx"
northsouth_velocity_url = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc2160_v/v_llc2160_x_y_depth.idx"

# Define quality level for balance
quality_level = -8  # Moderate resolution

try:
    # Load temperature dataset
    ds_temp = ovp.LoadDataset(temperature_url)
    ds_eastwest_vel = ovp.LoadDataset(eastwest_velocity_url)
    ds_northsouth_vel = ovp.LoadDataset(northsouth_velocity_url)
    
    # Extract data
    temperature_data = []
    eastwest_velocity_data = []
    northsouth_velocity_data = []
    
    for t in range(timestep_start, timestep_end, 720):  # Sample every 30 days (approx.)
        temp = ds_temp.db.read(
            time=t,
            x=x_range,
            y=y_range,
            z=[0, 1],  # Surface layer
            quality=quality_level
        )
        eastwest_vel = ds_eastwest_vel.db.read(
            time=t,
            x=x_range,
            y=y_range,
            z=[0, 1],  # Surface layer
            quality=quality_level
        )
        northsouth_vel = ds_northsouth_vel.db.read(
            time=t,
            x=x_range,
            y=y_range,
            z=[0, 1],  # Surface layer
            quality=quality_level
        )
        
        temperature_data.append(temp)
        eastwest_velocity_data.append(eastwest_vel)
        northsouth_velocity_data.append(northsouth_vel)
    
    # Save data
    np.savez(
        '/Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app/ai_data/data_cache/dyamond_llc2160/data_20251121_140122.npz',
        temperature=np.array(temperature_data),
        eastwest_velocity=np.array(eastwest_velocity_data),
        northsouth_velocity=np.array(northsouth_velocity_data),
        x_range=np.array(x_range),
        y_range=np.array(y_range),
        quality=quality_level
    )
    
    # Output summary
    print(json.dumps({
        "status": "success",
        "strategy": "Sampled every 30 days over 14 months in Agulhas region with moderate resolution",
        "data_points_processed": int(len(temperature_data) * np.prod(temp.shape)),
    }))
    
except Exception as e:
    print(json.dumps({"error": str(e)}))