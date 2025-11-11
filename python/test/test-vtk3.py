import numpy as np
import math
import os, sys
from OpenVisus import *
import argparse

# path to the rendering app libs
current_script_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.dirname(os.path.dirname(current_script_dir))
python_dir = os.path.join(base_path, 'python')
if python_dir not in sys.path:
    sys.path.append(python_dir)

build_path = os.path.join(base_path, 'build', 'renderingApps', 'py')
if build_path not in sys.path:
    sys.path.append(build_path)

import renderInterface3

parser = argparse.ArgumentParser(description='animation scripting tool examples')
parser.add_argument("-method", type=str, choices=["text", "viewer"],
                    help="choose scripting method")
parser.add_argument("-scene", type=str, choices=["flat", "sphere"],
                    help="choose scene")
parser.add_argument("-save", help="whether to download data to disk",
                    action="store_true")
parser.add_argument("-save_only", help="skip scripting and download data to disk",
                    action="store_true")

args = parser.parse_args()

# Data source URLs
eastwest_ocean_velocity_u = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc2160_arco/visus.idx"
northsouth_ocean_velocity_v = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc2160_v/v_llc2160_x_y_depth.idx"
vertical_velocity_w = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc2160_w/llc2160_w.idx"
temperature_theta = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc2160_theta/llc2160_theta.idx"
Salinity_salt = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc2160_salt/salt_llc2160_x_y_depth.idx"

def get_time_extent_from_datasrc(handler, logic_box=None):
    """Return (tmin, tmax, timesteps_list_or_None)."""
    tmin = None
    tmax = None
    timesteps = None

    try:
        db = getattr(handler.dataSrc, 'db', handler.dataSrc)
        if hasattr(db, 'getTimesteps'):
            timesteps = db.getTimesteps()
            try:
                timesteps = list(timesteps)
            except Exception:
                pass

            if isinstance(timesteps, (list, tuple)) and len(timesteps) > 0:
                first = timesteps[0]
                last = timesteps[-1]
                try:
                    tmin = int(first)
                    tmax = int(last)
                except Exception:
                    tmin = 0
                    tmax = len(timesteps) - 1
                return tmin, tmax, timesteps
    except Exception:
        timesteps = None

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
            if len(lower) >= 4 and len(upper) >= 4:
                tmin = int(lower[3])
                tmax = int(upper[3])
                return tmin, tmax, None
        except Exception:
            pass

    return None, None, None

def generate_adaptive_opacities(data_min, data_max, field_type="salinity", target_visibility="high"):
    """Generate adaptive opacity values based on data range to ensure streamline visibility"""
    data_range = data_max - data_min
    
    if data_range > 35:
        opacity_scale = 0.15
    elif data_range > 25:
        opacity_scale = 0.20
    elif data_range > 15:
        opacity_scale = 0.25
    else:
        opacity_scale = 0.35
    
    visibility_multiplier = {
        "low": 1.2,
        "medium": 1.0,
        "high": 0.7
    }[target_visibility]
    
    final_scale = opacity_scale * visibility_multiplier
    
    tf_opacities = [
        0.0,
        0.0,
        final_scale * 0.03,
        final_scale * 0.15,
        final_scale * 0.25,
        final_scale * 0.6,
        final_scale * 0.75,
        final_scale * 1.0,
        final_scale * 1.0
    ]
    
    print(f"Generated adaptive opacities for range [{data_min:.1f}, {data_max:.1f}]:")
    print(f"  Data range: {data_range:.1f}, Opacity scale: {final_scale:.3f}")
    print(f"  Max ocean opacity: {max(tf_opacities):.3f}")
    
    return tf_opacities

# Initialize handler
a = renderInterface3.AnimationHandler(Salinity_salt)

# Get data extents
box = a.dataSrc.getLogicBox()
try:
    if box is None and hasattr(a.dataSrc, 'db'):
        box = a.dataSrc.db.getLogicBox()
except Exception:
    try:
        box = a.dataSrc.db.getLogicBox()
    except Exception:
        box = None

print("raw logic box:", box)

if len(box) >= 2:
    lower = box[0]
    upper = box[1]
    try:
        xmin, ymin, zmin = lower[:3]
        xmax, ymax, zmax = upper[:3]
    except Exception:
        raise ValueError(f"Unexpected logic box shape: {box}")
    print(f"x range: 0 .. {xmax}   y range: 0 .. {ymax}   z range: 0 .. {zmax}")
else:
    raise ValueError(f"Invalid logic box returned: {box}")

# Set data dimensions
try:
    if box and len(box) >= 2:
        lower = box[0]
        upper = box[1]
        if len(upper) >= 3:
            xmax, ymax, zmax = int(upper[0]), int(upper[1]), int(upper[2])
        else:
            raise ValueError(f"Unexpected logic box upper bound: {upper}")
    else:
        raise ValueError("Logic box unavailable to derive x/y/z")

    tmin, tmax_from_helper, timesteps = get_time_extent_from_datasrc(a, logic_box=box)
    #print(f"Derived time extent from helper: tmin={tmin}, tmax={tmax_from_helper}, timesteps={timesteps}")
    
    if timesteps is not None:
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
    x_max = int(xmax)
    y_max = int(ymax)
    z_max = int(zmax)
    t_max = int(t_max_final)
except Exception as e:
    print(f"Could not set dims from dataset metadata: {e}. Falling back to hardcoded dims.")
    a.setDataDim(8640, 6480, 90, 10269)

# Define subregions
x_range = [1032, 2457]
y_range = [1872, 3141]
z_range = [0, 90]

times = np.linspace(24*91, 24*92, num=24, endpoint=False)

q = -12  # quality level


data = a.readData(t=times[0], x_range=x_range, y_range=y_range, z_range=z_range, q=q)
d_max = np.max(data)
d_min = np.min(data)

print(data.shape, d_max, d_min)

# Script configuration
input_names = a.getVTKFileNames(data.shape[2], data.shape[1], data.shape[0], times)
kf_interval = 1
dims = [data.shape[2], data.shape[1], data.shape[0]]
meshType = "streamline"
world_bbx_len = 10

base_output_dir = "/home/eliza89/PhD/codes/vis_user_tool/python/test/Out_text"
output_dirs = [os.path.join(base_output_dir, f) for f in input_names]

# VTK Camera format: position, focalPoint, up
def compute_camera_from_shape_q(data_shape, spacing=None, origin=None, subregion_offset=(0,0,0), force_pos=None, force_up=None, focal_from_center=False):
    """Compute a simple VTK camera (pos, focalPoint, up) from data.shape and q.

    - data_shape: numpy array shape returned by readData (z,y,x)
    - q: quality used to read the data (included for a slight heuristic scaling)
    Returns: cam list in VTK format [posx,posy,posz, fpx,fpy,fpz, upx,upy,upz]
    """
    nz, ny, nx = int(data_shape[0]), int(data_shape[1]), int(data_shape[2])
    print("compute_camera_from_shape_q: data_shape =", data_shape)
    # center in data index coordinates (0..N-1)
    cx = (nz - 1) / 2.0
    cy = (ny - 1) / 2.0
    cz = (nx - 1) / 2.0
    center_idx = [cx, cy, cz]

    # Option A: simple mode — compute orthographic camera focal from center index
    if focal_from_center:
        # center in index coordinates (use as focal point)
        center = [center_idx[0], center_idx[1], center_idx[2]]

        # If a forced position is provided, use it; otherwise compute fallback using diagonal
        if force_pos is not None:
            pos = [float(force_pos[0]), float(force_pos[1]), float(force_pos[2])]
        else:
            # compute diagonal in index-space (use spacing if provided to get world units)
            if spacing is None:
                sp = [1.0, 1.0, 1.0]
            else:
                sp = spacing
            diag = math.sqrt((nx * sp[0])**2 + (ny * sp[1])**2 + (nz * sp[2])**2)
            # Position rule: x = focal_x - diag, y = focal_y - diag, z = focal_z
            pos = [center[0] - diag, center[1] - diag, center[2]]

        # use forced up if provided, otherwise default to Y-up
        if force_up is not None:
            up = [float(force_up[0]), float(force_up[1]), float(force_up[2])]
        else:
            up = [0.0, 1.0, 0.0]

        # build camera in index-space units (as requested) and return early
        focal = center
        cam = [float(pos[0]), float(pos[1]), float(pos[2]),
               float(focal[0]), float(focal[1]), float(focal[2]),
               float(up[0]), float(up[1]), float(up[2])]
    
        return cam
    else:
        # map to world coordinates using spacing, origin and subregion offset if provided
        if spacing is None:
            spacing = [1.0, 1.0, 1.0]
        if origin is None:
            origin = [0.0, 0.0, 0.0]
        off_x, off_y, off_z = (int(subregion_offset[0]), int(subregion_offset[1]), int(subregion_offset[2])) if subregion_offset is not None else (0,0,0)

        # center in world coordinates: origin + (offset + center_index) * spacing
        center = [origin[0] + (off_x + center_idx[0]) * spacing[0],
                  origin[1] + (off_y + center_idx[1]) * spacing[1],
                  origin[2] + (off_z + center_idx[2]) * spacing[2]]
        pos = None
        up = None

    # For orthographic camera we compute focal point in world coords and place
    # the camera offset along X and Y by the scene diagonal (in world units).
    if spacing is None:
        spacing = [1.0, 1.0, 1.0]
    if origin is None:
        origin = [0.0, 0.0, 0.0]
    off_x, off_y, off_z = (int(subregion_offset[0]), int(subregion_offset[1]), int(subregion_offset[2])) if subregion_offset is not None else (0,0,0)

    # center in world coordinates: origin + (offset + center_index) * spacing
    center = [origin[0] + (off_x + center_idx[0]) * spacing[0],
              origin[1] + (off_y + center_idx[1]) * spacing[1],
              origin[2] + (off_z + center_idx[2]) * spacing[2]]

    # compute diagonal in world units
    diag = math.sqrt((nx * spacing[0])**2 + (ny * spacing[1])**2 + (nz * spacing[2])**2)

    # If a forced position is provided, use it; otherwise place camera at
    # [focal_x - diag, focal_y - diag, focal_z]
    if 'pos' in locals() and pos is not None:
        # already set earlier in focal_from_center branch
        pass
    else:
        pos = [center[0] - diag, center[1] - diag, center[2]]

    focal = center

    # up vector
    if 'up' in locals() and up is not None:
        # use provided up
        pass
    else:
        up = [0.0, 1.0, 0.0]

    cam = [float(pos[0]), float(pos[1]), float(pos[2]),
        float(focal[0]), float(focal[1]), float(focal[2]),
        float(up[0]), float(up[1]), float(up[2])]


    # also return the computed center in world coords/index coords for debugging if needed
    return cam

# compute camera automatically from returned data.shape and q
# Predict reduced dims from q and compute camera from predicted dims (so camera is stable)
def predicted_dims_from_q(q, original):
    """Predict returned dims (X,Y,Z) from quality q given original full-res dims.

    original: (X0, Y0, Z0)
    """
    X0, Y0, Z0 = original
    if q >= 0:
        return (int(X0), int(Y0), int(Z0))
    L = int(-q)
    base = L // 3
    r = L % 3
    ex_z = base + (1 if r >= 1 else 0)
    ex_y = base + (1 if r >= 2 else 0)
    ex_x = base
    # Use ceil as a reasonable prediction; server may floor/round to nearest tile
    X = int(math.ceil(float(X0) / (2 ** ex_x)))
    Y = int(math.ceil(float(Y0) / (2 ** ex_y)))
    Z = int(math.ceil(float(Z0) / (2 ** ex_z)))
    return (X, Y, Z)

# original full-resolution dims (use dataset metadata from logic box)
orig_full = (int(x_range[1]-x_range[0]), int(y_range[1]-y_range[0]), int(z_range[1]-z_range[0]))
pred_x, pred_y, pred_z = predicted_dims_from_q(q, orig_full)
print(f"Predicted dims for q={q}: x={pred_x}, y={pred_y}, z={pred_z}")

# compute camera from predicted dims (convert to data_shape ordering z,y,x)
pred_shape = (pred_z, pred_y, pred_x)
# swap_xz=True because renderer expects (z,y,x) world ordering
# For now: ignore spacing/origin/FOV — compute focal from predicted center and
# force position/up to your preferred values so the automatic calculation does not
# override your manual camera choice.
cam = compute_camera_from_shape_q(pred_shape, focal_from_center=True, force_pos=None, force_up=[0.0, 1.0, 0.0])
print("Computed camera from predicted dims:", cam)


# cam = [-300.0, -150.0, 145.0,  # position
#        11.0, 80.0, 145.0,       # focalPoint  
#        0.0, 1.0, 0.0]  for q= -6

tf_range = [d_min, d_max]

# Ocean water colors
tf_colors = [
0.0,
                        1.0,
                        1.0,
                        1.0,
                        0.125,
                        0.9330000281333923,
                        0.9570000171661377,
                        0.9800000190734863,
                        0.25,
                        0.8389999866485596,
                        0.8859999775886536,
                        0.9490000009536743,
                        0.375,
                        0.722000002861023,
                        0.8199999928474426,
                        0.8980000019073486,
                        0.5,
                        0.5529999732971191,
                        0.7179999947547913,
                        0.8429999947547913,
                        0.625,
                        0.3919999897480011,
                        0.6000000238418579,
                        0.7799999713897705,
                        0.75,
                        0.2590000033378601,
                        0.46299999952316284,
                        0.7059999704360962,
                        0.875,
                        0.15700000524520874,
                        0.3330000042915344,
                        0.7120000123977661,
                        1.0,
                        0.0860000029206276,
                        0.19200000166893005,
                        0.6200000047683716
]

tf_opacities = generate_adaptive_opacities(d_min, d_max, "salinity", "high")
scalar_field = "salinity"

# File size calculation (approximate)
file_sizes_mb = []
for i in range(len(times)):
    num_points = dims[0] * dims[1] * dims[2]
    bytes_per_point = 4
    size_mb = (num_points * bytes_per_point) / (1024 * 1024)
    file_sizes_mb.append(size_mb)

# Rendering configuration
frame_rate = 30.0
required_modules = [
    "vtkRenderingVolume",
    "vtkFiltersFlowPaths",
    "vtkRenderingCore",
    "vtkFiltersCore",
    "vtkCommonCore",
    "vtkIOXML"
]
grid_type = "structured"
spacing = [1.0, 1.0, 1.0]
origin = [0.0, 0.0, 0.0]
view_angle = 30.0
rendering_backend = "vtk"

# ==================================================================
# VOLUME REPRESENTATION CONFIG (Ocean volume rendering)
# ==================================================================
volume_representation_config = {
    "enabled": True,  # Enable volume rendering
    "volumeProperties": {
        "shadeOn": True,
        "interpolationType": "linear",
        "ambient": 0.2,
        "diffuse": 0.7,
        "specular": 0.1,
        "specularPower": 10.0,
        "scalarOpacityUnitDistance": 1.0,
        "independentComponents": True
    },
    "mapperProperties": {
        "type": "FixedPointVolumeRayCastMapper",
        "blendMode": "composite",
        "sampleDistance": 1.0,
        "autoAdjustSampleDistances": True,
        "imageSampleDistance": 1.0,
        "maximumImageSampleDistance": 10.0
    }
}

# ==================================================================
# STREAMLINE REPRESENTATION CONFIG (Velocity flow)
# ==================================================================
streamline_representation_config = {
    "enabled": True,  # Enable streamlines
    "integrationProperties": {
        "maxPropagation": 200,
        "initialIntegrationStep": 0.3,
        "minimumIntegrationStep": 0.05,
        "maximumIntegrationStep": 1.0,
        "integratorType": 2,
        "integrationDirection": "both",
        "maximumNumberOfSteps": 2000,
        "terminalSpeed": 1.0e-12,
        "computeVorticity": False,
        "rotationScale": 1.0,
        "surfaceStreamlines": False
    },
    "seedPlane": {
        "type": "plane",
        "enabled": True,
        "position": "quarter_x",
        "positionFraction": 0.25,
        "origin": None,
        "point1": None,
        "point2": None,
        "xResolution": 20,
        "yResolution": 20,
        "center": None,
        "normal": [1.0, 0.0, 0.0]
    },
    "seedPoints": {
        "type": "points",
        "enabled": False,
        "numberOfPoints": 100,
        "center": [0.0, 0.0, 0.0],
        "radius": 5.0,
        "distribution": "uniform",
        "points": []
    },
    "seedLine": {
        "type": "line",
        "enabled": False,
        "point1": [0.0, 0.0, 0.0],
        "point2": [10.0, 10.0, 10.0],
        "resolution": 50
    },
    "seedRake": {
        "type": "rake",
        "enabled": False,
        "startPoint": [0.0, 0.0, 0.0],
        "endPoint": [10.0, 0.0, 0.0],
        "numberOfLines": 10,
        "perpendicularDirection": [0.0, 1.0, 0.0],
        "lineLength": 5.0
    },
    "streamlineProperties": {
        "color": [1.0, 1.0, 1.0],
        "lineWidth": 1,
        "opacity": 1.0,
        "renderAsTubes": True,
        "tubeRadius": 0.1,
        "tubeNumberOfSides": 6,
        "tubeVaryRadius": "off",
        "tubeRadiusFactor": 10.0,
        "ambient": 0.3,
        "diffuse": 0.7,
        "specular": 0.1,
        "specularPower": 10,
        "specularColor": [1.0, 1.0, 1.0],
        "edgeVisibility": False,
        "edgeColor": [0.0, 0.0, 0.0],
        "lighting": True,
        "representation": "surface",
        "backfaceCulling": False,
        "frontfaceCulling": False
    },
    "colorMapping": {
        "colorByScalar": True,
        "scalarField": "salinity",
        "scalarMode": "usePointFieldData",
        "scalarRange": [34, 35.71],
        "autoRange": True,
        "colorSpace": "RGB",
        "nanColor": [0.5, 0.0, 0.0],
        "lookupTable": {
            "type": "preset",
            "presetName": "Rainbow",
            "customColors": [],
            "numberOfTableValues": 256,
            "hueRange": [0.667, 0.0],
            "saturationRange": [1.0, 1.0],
            "valueRange": [1.0, 1.0],
            "alphaRange": [1.0, 1.0],
            "scale": "linear",
            "ramp": "linear"
        }
    },
    "transferFunc": {
        "enabled": False,
        "range": [0.0, 1.5],
        "colors": [
            0.0, 0.0, 1.0,
            0.0, 1.0, 1.0,
            0.0, 1.0, 0.0,
            1.0, 1.0, 0.0,
            1.0, 0.0, 0.0
        ],
        "opacities": [0.5, 0.7, 0.9, 1.0, 1.0]
    },
    "outline": {
        "enabled": True,
        "color": [0.18431372549019609, 0.30980392156862744, 0.30980392156862744],
        "lineWidth": 0.3,
        "opacity": 1.0
    }
}

# ==================================================================
# ISOSURFACE REPRESENTATION CONFIG (Land mask)
# ==================================================================
isosurface_representation_config = {
    "enabled": True,  # Enable isosurface (land mask)
    "isoMethod": "threshold",
    "thresholdRange": [0.0, 0.005],
    "isoValues": [0.0025],
    "numberOfContours": 1,
    "computeNormals": True,
    "computeGradients": False,
    "computeScalars": True,
    "arrayComponent": 0,
    "surfaceProperties": {
        "color": [0.518, 0.408, 0.216],
        "opacity": 1.0,
        "ambient": 0.3,
        "diffuse": 0.7,
        "specular": 0.2,
        "specularPower": 20.0,
        "specularColor": [1.0, 1.0, 1.0],
        "metallic": 0.0,
        "roughness": 0.5,
        "edgeVisibility": False,
        "edgeColor": [0.0, 0.0, 0.0],
        "lighting": True,
        "interpolation": "gouraud",
        "representation": "surface",
        "backfaceCulling": False,
        "frontfaceCulling": False
    },
    "texture": {
        "enabled": True,
        "textureFile": "/home/eliza89/PhD/codes/vis_user_tool/renderingApps/vtkApps/agulhaas_mask_land.png",
        "mapMode": "plane",
        "repeat": False,
        "interpolate": True,
        "edgeClamp": False,
        "quality": "default",
        "blendMode": "replace",
        "transform": {
            "position": [0.0, 0.0],
            "scale": [1.0, 1.0],
            "rotation": 0.0
        }
    },
    "colorMapping": {
        "colorByScalar": False,
        "scalarField": "salinity",
        "scalarMode": "usePointFieldData",
        "scalarRange": [0.0, 0.005],
        "autoRange": False,
        "colorSpace": "RGB",
        "interpolateScalarsBeforeMapping": True
    },
    "transferFunc": {
        "enabled": False,
        "range": [0.0, 0.005],
        "colors": [
            0.518, 0.408, 0.216,
            0.618, 0.508, 0.316
        ],
        "opacities": [1.0, 1.0]
    }
}

# Generate script
if not args.save_only:
    gad_dir = os.path.join(current_script_dir, 'GAD_text')
    os.makedirs(gad_dir, exist_ok=True)
    outputName = os.path.join(gad_dir, "case2_script")
    
    # Call the new function with all 3 representation configs
    a.generateScriptStreamline(
        output_dirs,
        kf_interval,
        dims,
        meshType,
        world_bbx_len,
        cam,
        tf_range,
        tf_colors,
        tf_opacities,
        scalar_field,
        frame_rate,
        required_modules,
        file_sizes_mb,
        grid_type,
        spacing,
        origin,
        view_angle,
        rendering_backend,
        volume_representation_config,
        streamline_representation_config,
        isosurface_representation_config,
        outfile=outputName
    )
    
    print(f"Generated script with all 3 representations (volume, streamline, isosurface) at: {outputName}.json")

# Save data
# if args.save or args.save_only:
#     output_dir = os.path.join(current_script_dir, 'Out_text')
#     print("output_dir: ", output_dir)
#     os.makedirs(output_dir, exist_ok=True)
#     a.saveVTKFilesByVisusRead(
#         eastwest_ocean_velocity_u, 
#         northsouth_ocean_velocity_v, 
#         vertical_velocity_w, 
#         Salinity_salt, 
#         temperature_theta,
#         v_name="velocity",
#         active_scalar_name="salinity",
#         scalar2_name="temperature",
#         x_range=x_range, 
#         y_range=y_range, 
#         z_range=z_range, 
#         t_list=times, 
#         q=q, 
#         flip_axis=2, 
#         transpose=False, 
#         output_dir=output_dir
#     )
#     print("Data saved to disk with velocity vectors for eddy flow arrows")