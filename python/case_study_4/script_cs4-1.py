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


a = renderInterface.AnimationHandler("https://maritime.sealstorage.io/api/v0/s3/utah/nasa/dyamond/mit_output/llc2160_theta/llc2160_theta.idx?access_key=any&secret_key=any&endpoint_url=https://maritime.sealstorage.io/api/v0/s3&cached=arco")

# set render info
a.setDataDim(8640, 6480, 90, 10269);

#
# choose data region
# 
x_range = [0, int(a.x_max)]
y_range = [0, int(a.y_max)]
z_range = [0, a.z_max]
t_list = np.linspace(0, 24*365, num=20, endpoint=False);
quality=-12

data = a.readData(t=t_list[0], x_range=x_range, y_range=y_range,z_range=z_range, q=quality, flip_axis=2, transpose=False)

dim = data.shape
d_max = np.max(data)
d_min = np.min(data)
print(dim, d_max, d_min)

# set script details
input_names = a.getRawFileNames(data.shape[2], data.shape[1], data.shape[0], t_list)
kf_interval = 1 # frames per keyframe
dims = [data.shape[2], data.shape[1], data.shape[0]] # flip axis from py array
meshType = "structuredSpherical"
cam = [-30, 0, 0, 1, 0, 0, 0, 0, -1]
world_bbx_len = 10
tf_range = [d_min, d_max]

# generate script
gad_dir = os.path.join(current_script_dir, 'GAD_text')
os.makedirs(gad_dir, exist_ok=True)
outputName = os.path.join(gad_dir, "case4_script")

bgImg = os.path.join(base_path, 'renderingApps', 'mesh', 'land.png')

a.generateScript(input_names, kf_interval, dims, meshType, world_bbx_len, cam,  tf_range, template="rotate", s=45, e=135, dist=25, outfile=outputName, bgImg=bgImg);


if (args.save or args.save_only):
    output_dir = os.path.join(current_script_dir, 'Out_text')
    print("output_dir: ", output_dir)
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    a.saveRawFilesByVisusRead(t_list=t_list, x_range=x_range, y_range=y_range,z_range=z_range, q=quality, flip_axis=2, transpose=False, output_dir=output_dir)
    print("Data saved to disk")



# For a smooth orbit around the Earth, use these 8 camera positions:

# 1. Position: [-30, 0, 0], Direction: [1, 0, 0]
# 2. Position: [-21.21, -21.21, 0], Direction: [0.7071, 0.7071, 0]
# 3. Position: [0, -30, 0], Direction: [0, 1, 0]
# 4. Position: [21.21, -21.21, 0], Direction: [-0.7071, 0.7071, 0]
# 5. Position: [30, 0, 0], Direction: [-1, 0, 0]
# 6. Position: [21.21, 21.21, 0], Direction: [-0.7071, -0.7071, 0]
# 7. Position: [0, 30, 0], Direction: [0, -1, 0]
# 8. Position: [-21.21, 21.21, 0], Direction: [0.7071, -0.7071, 0]

