import numpy as np
from OpenVisus import *
import argparse

# path to the rendering app libs
current_script_dir = os.path.dirname(os.path.abspath(__file__))  # .../python/case_study_1-5

# Compute the base path (two levels up from the script's directory)
base_path = os.path.dirname(os.path.dirname(current_script_dir))  # .../vis_user_tool

# Add the `python/` directory to find renderInterface.py
python_dir = os.path.join(base_path, 'python')
if python_dir not in sys.path:
    sys.path.append(python_dir)

# Add the `build/renderingApps/py` directory to find vistool_py
build_path = os.path.join(base_path, 'build', 'renderingApps', 'py')
if build_path not in sys.path:
    sys.path.append(build_path)

import renderInterface

parser = argparse.ArgumentParser(description='animation scripting tool examples')
parser.add_argument("-method", type=str, choices=["text", "viewer"],
                    help="choose scripting method")
parser.add_argument("-scene", type=str, choices=["flat", "sphere"],
                    help="choose scene")
parser.add_argument("-save", help="whether to download data to disk",
                    action="store_true")
parser.add_argument("-save_only", help="skip scripting and download data to disk",
                    action="store_true")

args=parser.parse_args()

# eastwest_ocean_velocity_u="pelican://osg-htc.org/nasa/nsdf/climate3/dyamond/mit_output/llc2160_arco/visus.idx"
# northsouth_ocean_velocity_v="pelican://osg-htc.org/nasa/nsdf/climate3/dyamond/mit_output/llc2160_v/v_llc2160_x_y_depth.idx"
# vertical_velocity_w="pelican://osg-htc.org/nasa/nsdf/climate3/dyamond/mit_output/llc2160_w/llc2160_w.idx"
# temperature_theta="pelican://osg-htc.org/nasa/nsdf/climate3/dyamond/mit_output/llc2160_theta/llc2160_theta.idx"
# Salinity_salt = "pelican://osg-htc.org/nasa/nsdf/climate3/dyamond/mit_output/llc2160_salt/salt_llc2160_x_y_depth.idx"eastwest_ocean_velocity_u="pelican://osg-htc.org/nasa/nsdf/climate3/dyamond/mit_output/llc2160_arco/visus.idx"
eastwest_ocean_velocity_u="https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc2160_arco/visus.idx"
northsouth_ocean_velocity_v="https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc2160_v/v_llc2160_x_y_depth.idx"
vertical_velocity_w="https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc2160_w/llc2160_w.idx"
temperature_theta="https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc2160_theta/llc2160_theta.idx"
Salinity_salt = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc2160_salt/salt_llc2160_x_y_depth.idx"

# Unpack velocity information


a = renderInterface.AnimationHandler(Salinity_salt)
# ---- time extent helper (paste into test-vtk.py) ----
def get_time_extent_from_datasrc(handler, logic_box=None):
    """
    Return (tmin, tmax, timesteps_list_or_None).
    - timesteps_list is returned if available (as Python list).
    - tmin/tmax are integers when possible; otherwise None.
    """
    tmin = None
    tmax = None
    timesteps = None

    # 1) Try explicit timesteps from the DB wrapper (preferred)
    try:
        db = getattr(handler.dataSrc, 'db', handler.dataSrc)
        if hasattr(db, 'getTimesteps'):
            timesteps = db.getTimesteps()
            # normalize to list
            try:
                timesteps = list(timesteps)
            except Exception:
                # already a list-like or cannot convert
                pass

            if isinstance(timesteps, (list, tuple)) and len(timesteps) > 0:
                # Many backends return either integer indices or actual time values.
                # If values look like integer indices, use them directly.
                first = timesteps[0]
                last = timesteps[-1]
                try:
                    # try integer interpretation
                    tmin = int(first)
                    tmax = int(last)
                except Exception:
                    # if non-integer (e.g., datetimes or floats), fall back to index range
                    tmin = 0
                    tmax = len(timesteps) - 1
                return tmin, tmax, timesteps
    except Exception:
        timesteps = None

    # 2) Fall back to logic box 4th component if present
    if logic_box is None:
        try:
            logic_box = handler.dataSrc.getLogicBox()
        except Exception:
            try:
                logic_box = handler.dataSrc.db.getLogicBox()
            except Exception:
                logic_box = None

    if logic_box:
        try:
            lower = logic_box[0]
            upper = logic_box[1]
            # Some logic boxes are (x,y,z) only; others include time as a 4th component.
            if len(lower) >= 4 and len(upper) >= 4:
                tmin = int(lower[3])
                tmax = int(upper[3])
                return tmin, tmax, None
        except Exception:
            pass

    # 3) No time info found
    return None, None, None


# (Example usage moved below, after we fetch the logic box.)
box = a.dataSrc.getLogicBox() # many code sites use this form
try:
    # some wrappers expose .db.getLogicBox(); try that if necessary
    if box is None and hasattr(a.dataSrc, 'db'):
        box = a.dataSrc.db.getLogicBox()
except Exception:
    try:
        box = a.dataSrc.db.getLogicBox()
    except Exception:
        box = None

print("raw logic box:", box) 
# Support both 3-element (x,y,z) and 4-element (x,y,z,t) logic boxes
if len(box) >= 2:
    lower = box[0]
    upper = box[1]
    # Always extract x/y/z components; ignore time here (we derive time separately)
    try:
        xmin, ymin, zmin = lower[:3]
        xmax, ymax, zmax = upper[:3]
    except Exception:
        raise ValueError(f"Unexpected logic box shape: {box}")

    print(f"x range: 0 .. {xmax}   y range: 0 .. {ymax}   z range: 0 .. {zmax}")
else:
    raise ValueError(f"Invalid logic box returned: {box}")



# Determine dataset dims and set them on the handler (prefer metadata over hardcoding)
try:
    # parse logic box for x/y/z
    if box and len(box) >= 2:
        lower = box[0]
        upper = box[1]
        if len(upper) >= 3:
            xmax, ymax, zmax = int(upper[0]), int(upper[1]), int(upper[2])
        else:
            raise ValueError(f"Unexpected logic box upper bound: {upper}")
    else:
        raise ValueError("Logic box unavailable to derive x/y/z")

    # Derive time extent using helper (timesteps preferred)
    tmin, tmax_from_helper, timesteps = get_time_extent_from_datasrc(a, logic_box=box)
    # print(f"Derived time extent from helper: tmin={tmin}, tmax={tmax_from_helper}, timesteps={timesteps}")
    if timesteps is not None:
        # use last index as t_max (if timesteps are indices or numeric timestamps)
        try:
            t_max_final = int(timesteps[-1])
        except Exception:
            t_max_final = len(timesteps) - 1
    elif tmax_from_helper is not None:
        t_max_final = int(tmax_from_helper)
    else:
        t_max_final = 0

    print(f"Setting handler data dims from dataset: x={xmax}, y={ymax}, z={zmax}, t={t_max_final}")
    a.setDataDim(xmax, ymax, zmax, t_max_final)
    # Cache common extent values so we don't repeatedly query the data source
    x_max = int(xmax)
    y_max = int(ymax)
    z_max = int(zmax)
    t_max = int(t_max_final)
except Exception as e:
    print(f"Could not set dims from dataset metadata: {e}. Falling back to hardcoded dims.")
    a.setDataDim(8640, 6480, 90, 10269)

# reuse cached values computed above
# x_max, y_max, z_max, t_max were set when we called setDataDim
# Use those cached integers instead of querying the data source again

# Choose subregions as fractions of the full dataset extents
x_range = [int(x_max * 0.119), int(x_max * 0.253)]
y_range = [int(y_max * 0.378667), int(y_max * 0.501333)]

# x_range = [int(x_max*0.057), int(x_max*0.174)]
# y_range = [int(y_max*0.69), int(y_max*0.802)]

# x_range = [int(x_max*0.153), int(x_max*0.289)]
# y_range = [int(y_max*0.645), int(y_max*0.735)]

z_range = [0, z_max]

times = np.linspace(24*91, 24*92, num=24, endpoint=False);
q=-6

data = a.readData(t=times[0], x_range=x_range, y_range=y_range,z_range=z_range, q=q)
d_max = np.max(data)
d_min = np.min(data)

print(data.shape, d_max, d_min)



# set script details
input_names = a.getVTKFileNames(data.shape[2], data.shape[1], data.shape[0], times)
kf_interval = 1 # frames per keyframe
dims = [data.shape[2], data.shape[1], data.shape[0]] # flip axis from py array
meshType = "streamline"
world_bbx_len = 10
cam = [-10, -5, 17, 0.56, 0.42, -0.71, -0.7, 0.43, -0.46] # camera params, pos, dir, up

# cam = [4.735267639160156, -3.0468220710754395, -6.374855995178223, 0, 0.6900219321250916, 0.7233046889305115, 0, 0.7236120104789734, -0.6902016997337341] # camera params, pos, dir, up
tf_range = [d_min, d_max]

# Define colors and opacities for transfer function - Ocean water colors
# These will be mapped based on the data range [d_min, d_max] for ocean volume rendering
tf_colors = [
    1.0, 1.0, 1.0,     # White for land (will be made transparent)
    0.933, 0.957, 0.980,  # Very light blue
    0.839, 0.886, 0.949,  # Light blue  
    0.722, 0.820, 0.898,  # Medium light blue
    0.553, 0.718, 0.843,  # Medium blue
    0.392, 0.600, 0.780,  # Darker blue
    0.259, 0.463, 0.706,  # Dark blue
    0.157, 0.333, 0.712,  # Very dark blue
    0.086, 0.192, 0.620   # Deepest blue
]

def generate_adaptive_opacities(data_min, data_max, field_type="salinity", target_visibility="high"):
    """
    Generate adaptive opacity values based on data range to ensure streamline visibility
    """
    data_range = data_max - data_min
    
    # Adaptive opacity scaling based on data range
    # Larger ranges need lower opacity to prevent volume accumulation
    if data_range > 35:
        opacity_scale = 0.15  # Very transparent for large ranges
    elif data_range > 25:
        opacity_scale = 0.20  # Moderately transparent
    elif data_range > 15:
        opacity_scale = 0.25  # Standard transparency
    else:
        opacity_scale = 0.35  # Higher opacity for small ranges
    
    # Adjust based on target visibility
    visibility_multiplier = {
        "low": 1.2,      # More opaque volume
        "medium": 1.0,   # Standard
        "high": 0.7      # More transparent for better streamlines
    }[target_visibility]
    
    final_scale = opacity_scale * visibility_multiplier
    
    # Generate adaptive opacity curve
    tf_opacities = [
        0.0,                      # Transparent land
        0.0,                      # Transparent land boundary  
        final_scale * 0.03,       # Very transparent coastal water
        final_scale * 0.15,       # Very transparent low salinity ocean
        final_scale * 0.25,       # Still very transparent medium salinity
        final_scale * 0.6,        # Slightly more visible high salinity
        final_scale * 0.75,       # Semi-transparent very high salinity
        final_scale * 1.0,        # Maximum transparency at max salinity
        final_scale * 1.0         # Ensure enough points
    ]
    
    print(f"Generated adaptive opacities for range [{data_min:.1f}, {data_max:.1f}]:")
    print(f"  Data range: {data_range:.1f}, Opacity scale: {final_scale:.3f}")
    print(f"  Max ocean opacity: {max(tf_opacities):.3f}")
    
    return tf_opacities

# Ocean opacity values - land transparent, ocean semi-transparent
# tf_opacities = [
#     0.0,   # Transparent land
#     0.0,   # Transparent land boundary  
#     0.01,  # Very transparent coastal water
#     0.05,  # Very transparent low salinity ocean
#     0.08,  # Still very transparent medium salinity
#     0.2,   # Slightly more visible high salinity
#     0.24,  # Semi-transparent very high salinity
#     0.33,  # Maximum transparency at max salinity
#     0.33   # Maximum transparency at max range
# ]

tf_opacities = generate_adaptive_opacities(d_min, d_max, "salinity", "high")

# Define scalar field name based on the data source
scalar_field = "salinity"  # This corresponds to salinity data source

if (args.save_only == False):
	# generate script
    gad_dir = os.path.join(current_script_dir, 'GAD_text')
    os.makedirs(gad_dir, exist_ok=True)
    outputName = os.path.join(gad_dir, "case2_script")
    a.generateScriptStreamline(input_names, kf_interval, dims, meshType, world_bbx_len, cam, tf_range, tf_colors, tf_opacities, scalar_field, template="fixedCam", outfile=outputName);

if (args.save or args.save_only):
    output_dir = os.path.join(current_script_dir, 'Out_text')
    print("output_dir: ", output_dir)
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Save VTK files with velocity vectors for proper eddy flow arrows
    a.saveVTKFilesByVisusRead(eastwest_ocean_velocity_u, northsouth_ocean_velocity_v, vertical_velocity_w, Salinity_salt, t_list=times, x_range=x_range, y_range=y_range,z_range=z_range, q=q, flip_axis=2, transpose=False, output_dir=output_dir)
    print("Data saved to disk with velocity vectors for eddy flow arrows")

