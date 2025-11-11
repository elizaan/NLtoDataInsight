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

#a = renderInterface.AnimationHandler("https://maritime.sealstorage.io/api/v0/s3/utah/nasa/dyamond/mit_output/llc2160_theta/llc2160_theta.idx?access_key=any&secret_key=any&endpoint_url=https://maritime.sealstorage.io/api/v0/s3&cached=arco")
a = renderInterface.AnimationHandler("https://maritime.sealstorage.io/api/v0/s3/utah/nasa/dyamond/mit_output/llc2160_theta/llc2160_theta.idx?access_key=any&secret_key=any&endpoint_url=https://maritime.sealstorage.io/api/v0/s3&cached=arco")

# set render info
a.setDataDim(8640, 6480, 90, 10269);

#
# choose data region, if applicable
#
# For Indian Ocean Dipole (IOD)
x_max = a.dataSrc.getLogicBox()[1][0]
y_max = a.dataSrc.getLogicBox()[1][1]

# This range captures both the western Indian Ocean (near East Africa) 
# and eastern Indian Ocean (near Indonesia/Australia)
x_range = [int(x_max*0.18), int(x_max*0.35)]  # Covers longitude from ~50°E to ~100°E
y_range = [int(y_max*0.40), int(y_max*0.60)]  # Covers latitude from ~10°S to ~15°N
z_range = [0, 30] 
# Since IOD is a seasonal phenomenon, use a wider time range to capture its evolution
# Ideally spanning several months during the typical IOD peak (September-November)
t_list = np.arange(6000, 6500, 24, dtype=int).tolist()
quality=-8

#
# scripting
#

scriptingType = args.method if args.method else "text"
testing_scene = args.scene if args.scene else "flat"


gad_text_dir = os.path.join(current_script_dir, 'GAD_text')
gad_viewer_dir = os.path.join(current_script_dir, 'GAD_viewer')
out_text_dir = os.path.join(current_script_dir, 'Out_text')
out_viewer_dir = os.path.join(current_script_dir, 'Out_viewer')

# Create directories
os.makedirs(gad_text_dir, exist_ok=True)
os.makedirs(gad_viewer_dir, exist_ok=True)
if args.save or args.save_only:
    os.makedirs(out_text_dir, exist_ok=True)
    os.makedirs(out_viewer_dir, exist_ok=True)

# Set output file names
outputName_text = os.path.join(gad_text_dir, "text_script")
outputName_viewer = os.path.join(gad_viewer_dir, "viewer_script")

# additional options:
# if the data need to be flipped or transposed 
flip_axis=2
transpose=False
bgImg = ''

if(testing_scene=="flat"):
    flip_axis=2
    transpose=False
    render_mode=0
    bgImg = ''
elif(testing_scene=="sphere"):
    flip_axis=2
    transpose=True
    render_mode=2
    bgImg = os.path.join(base_path, 'renderingApps', 'mesh', 'land.png')

if (args.save_only==False):
    # produce scripts from one of
    if (scriptingType == "viewer"):
        # 1. interactive app
        print("Launching interactive viewer")
        print(f"Viewer script will be saved to: {outputName_viewer}")

        os.environ["KF_WIDGET_OUTPUT_PATH"] = outputName_viewer
        
        # Start the interactive viewer (saving directly to GAD_viewer)
        viewer_thread = Thread(target=a.renderTask, kwargs={
            't_list': t_list,
            'x_range': x_range,
            'y_range': y_range,
            'z_range': z_range,
            'q': quality,
            'mode': render_mode,
            'flip_axis': flip_axis,
            'transpose': transpose,
            'bgImg': bgImg,
            'outputName': outputName_viewer
        })
        viewer_thread.start()

        viewer_thread.join()
    
    elif (scriptingType == "text"):
        # 2. text scripting
        # generate key frames from scripting templates
        # i.e. generate animation w/ fixed cam all for all timesteps
        
        # read one timestep for data stats
        print("Generating text script")
        data = a.readData(t=t_list[0], x_range=x_range, y_range=y_range,z_range=z_range, q=quality, flip_axis=flip_axis, transpose=transpose)
        dim = data.shape
        d_max = np.max(data)
        d_min = np.min(data)
        print(dim, d_max, d_min)
        
        # set script details
        script_template="fixedCam"
        input_names = a.getRawFileNames(data.shape[2], data.shape[1], data.shape[0], t_list)
        kf_interval = 1 # frames per keyframe
        dims = [data.shape[2], data.shape[1], data.shape[0]] # flip axis from py array
        meshType = "structured"
        world_bbx_len = 10
        cam = [5, 4, -10, 0, 0, 1, 0, 1, 0] # camera params, pos, dir, up
        tf_range = [d_min, d_max]
        
        if(testing_scene=="flat"):
            meshType = "structured"
            cam = [5, 4, -10, 0, 0, 1, 0, 1, 0]
        elif(testing_scene=="sphere"):
            meshType = "structuredSpherical"
            cam = [-30, 0, 0, 1, 0, 0, 0, 0, -1]
            
        # generate script
        a.generateScript(input_names, kf_interval, dims, meshType, world_bbx_len, cam, tf_range, template=script_template, outfile=outputName_text, bgImg=bgImg);

#
# download data for offline render
#

if (args.save or args.save_only):
    # Determine which output directory to use based on the scripting type
    output_dir = out_viewer_dir if scriptingType == "viewer" else out_text_dir
    print(f"Raw data output_dir: {output_dir}")
    
    # Save the raw files
    a.saveRawFilesByVisusRead(t_list=t_list, x_range=x_range, y_range=y_range, z_range=z_range, q=quality, flip_axis=flip_axis, transpose=transpose, output_dir=output_dir)