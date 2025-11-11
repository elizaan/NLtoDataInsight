import sys, os
import numpy as np
from threading import Thread
import argparse

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

#
# set database source
#

a = renderInterface.AnimationHandler("https://maritime.sealstorage.io/api/v0/s3/utah/nasa/dyamond/mit_output/llc2160_theta/llc2160_theta.idx?access_key=any&secret_key=any&endpoint_url=https://maritime.sealstorage.io/api/v0/s3&cached=arco")

# set render info
a.setDataDim(8640, 6480, 90, 10269);

#
# choose data region
# 

# Agulhas Current
x_range = [int(a.x_max*0.119), int(a.x_max*0.253)]
y_range = [int(a.y_max*0.378667), int(a.y_max*0.501333)]
z_range = [0, a.z_max]
t_list = np.arange(0, 40, 4, dtype=int).tolist()
quality=-6 #why not 0


#
# scripting
#

# read one timestep for data stats

data = a.readData(t=t_list[0], x_range=x_range, y_range=y_range,z_range=z_range, q=quality, flip_axis=2, transpose=False)

dim = data.shape
d_max = np.max(data)
d_min = np.min(data)
print(dim, d_max, d_min)
        
# set script details
input_names = a.getRawFileNames(data.shape[2], data.shape[1], data.shape[0], t_list)
kf_interval = 1 # frames per keyframe
dims = [data.shape[2], data.shape[1], data.shape[0]] # flip axis from py array
meshType = "structured"
world_bbx_len = 10
cam = [4.735267639160156, -3.0468220710754395, -6.374855995178223, 0, 0.6900219321250916, 0.7233046889305115, 0, 0.7236120104789734, -0.6902016997337341] # camera params, pos, dir, up
tf_range = [d_min, d_max]
meshType = "structured"

# generate script
gad_dir = os.path.join(current_script_dir, 'GAD_text')
os.makedirs(gad_dir, exist_ok=True)
outputName = os.path.join(gad_dir, "case1_script")

a.generateScript(input_names, kf_interval, dims, meshType, world_bbx_len, cam, tf_range, template="fixedCam", outfile=outputName);

#
# download data for offline render
#


if (args.save or args.save_only):
    output_dir = os.path.join(current_script_dir, 'Out_text')
    print("output_dir: ", output_dir)
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    a.saveRawFilesByVisusRead(t_list=t_list, x_range=x_range, y_range=y_range,z_range=z_range, q=quality, flip_axis=2, transpose=False, output_dir=output_dir)

