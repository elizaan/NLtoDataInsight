import sys, os
import numpy as np
from threading import Thread
import argparse
import requests
from urllib.parse import urlparse

# path to the rendering app libs
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'build', 'renderingApps', 'py'))
print(os.path.join(os.path.dirname(sys.path[0]), 'build', 'renderingApps', 'py'))
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

a = renderInterface.AnimationHandler("pelican://osg-htc.org/nasa/nsdf/climate3/dyamond/mit_output/llc2160_theta/llc2160_theta.idx")

#a = renderInterface.AnimationHandler("https://maritime.sealstorage.io/api/v0/s3/utah/nasa/dyamond/mit_output/llc2160_salt/salt_llc2160_x_y_depth.idx?access_key=any&secret_key=any&endpoint_url=https://maritime.sealstorage.io/api/v0/s3&cached=arco")

# set render info
a.setDataDim(8640, 6480, 90, 10269);

#
# choose data region, if applicable
#

# Agulhas Current
#   "x_range": [int(self.animation.x_max*0.023), int(self.animation.x_max*0.138)],
#                 "y_range": [int(self.animation.y_max*0.69), int(self.animation.y_max*0.82)],
# mediterranean sea
# x_range = [int(a.x_max*0.027), int(a.x_max*0.137)]
# y_range = [int(a.y_max*0.69), int(a.y_max*0.82)]
# x_range = [int(a.x_max*0.119), int(a.x_max*0.253)]
# y_range = [int(a.y_max*0.378667), int(a.y_max*0.501333)]
# Agulhas Current
x_range = [int(a.x_max*0.119), int(a.x_max*0.253)]
y_range = [int(a.y_max*0.378667), int(a.y_max*0.501333)]
# x_range = [7000, 8640]
# y_range = [4000, 5400]
# x_range = [233, 1192]
# y_range = [4471, 5313]
z_range = [0, a.z_max]
# x_range = [0, 8640]
# y_range = [0, 6480]
# z_range = [0, a.z_max]
t_list = np.arange(0, 24*9, 24, dtype=int).tolist()
quality=-12

#
# scripting
#

scriptingType = args.method if args.method else "text"
testing_scene = args.scene if args.scene else "flat"

# Setup directories for both methods
base_dir = os.path.dirname(sys.path[0])
gad_text_dir = os.path.join(base_dir, 'GAD_text')
gad_viewer_dir = os.path.join(base_dir, 'GAD_viewer')
out_text_dir = os.path.join(base_dir, 'Out_text')
out_viewer_dir = os.path.join(base_dir, 'Out_viewer')

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
    bgImg = os.path.join(base_dir, 'renderingApps', 'mesh', 'land.png')

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
        # cam = [4.735267639160156, -3.0468220710754395, -6.374855995178223, 0, 0.6900219321250916, 0.7233046889305115, 0, 0.7236120104789734, -0.6902016997337341] # camera params, pos, dir, up
        # cam =  [4.84619, -3.10767, -8.16708, 0.0138851, 0.607355, 0.794303, -0.00317818, 0.794402, -0.607375]
        # cam = [-10, -5, 20, 0.56, 0.42, -0.71, 0, 0.43, -0.46]
        # cam =[-3.83565, 0.829617, -8.0832, 0.638209, 0.204561, 0.742186, 0.368159, 0.765575, -0.527589]
        # cam = [-3.81965, -2.3812, -5.92815, 0.684406, 0.504159, 0.526698, 0.213723, 0.551935, -0.806033]
        # cam = [-3.74447, -2.11408, -6.84755, 0.622472, 0.436486, 0.649624, 0.355763, 0.58152, -0.731621]
        tf_range = [d_min, d_max]
        
        if(testing_scene=="flat"):
            meshType = "structured"
            #cam =  [4.84619, -3.10767, -8.16708, 0.0138851, 0.607355, 0.794303, -0.00317818, 0.794402, -0.607375] # camera params, pos, dir, up
            cam = [4.96626, -3.56828, -8.20303, 0.00169525, 0.674153, 0.738601, 0.00813293, 0.738569, -0.674142]
            # mediterranean sea salinity to match vtk
            # cam =[-3.83565, 0.829617, -8.0832, 0.638209, 0.204561, 0.742186, 0.368159, 0.765575, -0.527589]
            # cam = [-2.54826, -1.28879, -5.18024, 0.650855, 0.252346, 0.716042, 0.338019, 0.748195, -0.570924]
            #ncam = [-2.0963, -1.69889, -7.34004, 0.539392, 0.170528, 0.824622, 0.369508, 0.83204, -0.41376]
            # cam = [-3.74447, -2.11408, -6.84755, 0.622472, 0.436486, 0.649624, 0.355763, 0.58152, -0.731621]
            # cam = [-3.81965, -2.3812, -5.92815, 0.684406, 0.504159, 0.526698, 0.213723, 0.551935, -0.806033]
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