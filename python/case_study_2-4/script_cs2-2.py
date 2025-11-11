import numpy as np
# import pyvista as pv
from matplotlib.colors import ListedColormap
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


eastwest_ocean_velocity_u="https://maritime.sealstorage.io/api/v0/s3/utah/nasa/dyamond/mit_output/llc2160_arco/visus.idx?access_key=any&secret_key=any&endpoint_url=https://maritime.sealstorage.io/api/v0/s3&cached=arco"
northsouth_ocean_velocity_v="https://maritime.sealstorage.io/api/v0/s3/utah/nasa/dyamond/mit_output/llc2160_v/v_llc2160_x_y_depth.idx?access_key=any&secret_key=any&endpoint_url=https://maritime.sealstorage.io/api/v0/s3&cached=arco"
vertical_velocity_w="https://maritime.sealstorage.io/api/v0/s3/utah/nasa/dyamond/mit_output/llc2160_w/llc2160_w.idx?access_key=any&secret_key=any&endpoint_url=https://maritime.sealstorage.io/api/v0/s3&cached=arco"
temperature_theta="https://maritime.sealstorage.io/api/v0/s3/utah/nasa/dyamond/mit_output/llc2160_theta/llc2160_theta.idx?access_key=any&secret_key=any&endpoint_url=https://maritime.sealstorage.io/api/v0/s3&cached=arco"

Salinity_salt="https://maritime.sealstorage.io/api/v0/s3/utah/nasa/dyamond/mit_output/llc2160_salt/salt_llc2160_x_y_depth.idx?access_key=any&secret_key=any&endpoint_url=https://maritime.sealstorage.io/api/v0/s3&cached=arco"

# Unpack velocity information


a = renderInterface.AnimationHandler(Salinity_salt)

# set render info
a.setDataDim(8640, 6480, 90, 10269);

x_max = a.dataSrc.getLogicBox()[1][0]
y_max = a.dataSrc.getLogicBox()[1][1]
x_range = [int(x_max*0.057), int(x_max*0.134)]
y_range = [int(y_max*0.69), int(y_max*0.81)]
z_range = [0, a.z_max]

times = np.linspace(0, 24*30, num=2, endpoint=False);
q=-8

data = a.readData(t=times[0], x_range=x_range, y_range=y_range,z_range=z_range, q=q)
d_max = np.max(data)
d_min = np.min(data)

print(f"Data shape: {data.shape}, Min: {d_min}, Max: {d_max}")
# set script details
input_names = a.getVTKFileNames(data.shape[2], data.shape[1], data.shape[0], times)
kf_interval = 1 # frames per keyframe
dims = [data.shape[2], data.shape[1], data.shape[0]] # flip axis from py array
meshType = "streamline"
world_bbx_len = 10
# cam = [-10, -2, 17, 0.56, 0.42, -0.71, -0.7, 0.43, -0.46] #rendered frames 1
cam = [-10, -5, 20, 0.56, 0.42, -0.71, 0, 0.43, -0.46] # camera params, pos, dir, up renderedframes 2




# cam = [-5, -10, 25, 0.2, 0.5, -0.8, 0, 1, 0.1] # overhead
# cam = [-15, -15, 20, 0.6, 0.6, -0.5, 0, 0, 1] # diagonal
# cam = [0, -25, 10, 0, 0.9, 0, 0, 0, 1] # front
tf_range = [d_min, d_max]

if (args.save_only == False):
	# generate script
    gad_dir = os.path.join(current_script_dir, 'GAD_text')
    os.makedirs(gad_dir, exist_ok=True)
    outputName = os.path.join(gad_dir, "case2_script")
    a.generateScriptStreamline(input_names, kf_interval, dims, meshType, world_bbx_len, cam, tf_range, template="fixedCam", outfile=outputName);

if (args.save or args.save_only):
    output_dir = os.path.join(current_script_dir, 'Out_text')
    print("output_dir: ", output_dir)
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    a.saveVTKFilesByVisusRead(eastwest_ocean_velocity_u, northsouth_ocean_velocity_v, vertical_velocity_w, Salinity_salt, t_list=times, x_range=x_range, y_range=y_range,z_range=z_range, q=q, flip_axis=2, transpose=False, output_dir=output_dir)
    print("Data saved to disk")

