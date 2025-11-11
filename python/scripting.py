import sys, os
import numpy as np
from threading import Thread
import argparse

# path to the rendering app libs
# print(os.path.join(os.path.dirname(sys.path[0]), 'build', 'renderingApps', 'py'))

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

a = renderInterface.AnimationHandler("https://maritime.sealstorage.io/api/v0/s3/utah/nasa/dyamond/mit_output/llc2160_theta/llc2160_theta.idx?access_key=any&secret_key=any&endpoint_url=https://maritime.sealstorage.io/api/v0/s3&cached=arco")

# set render info
a.setDataDim(8640, 6480, 90, 10269);

#
# choose data region, if applicable
#

# global
#t_list = np.arange(0, 1000, 2400, dtype=int).tolist()
#x_range = [0, a.x_max]
#y_range = [0, a.y_max]
#z_range = [0, a.z_max]
#quality=-6

# Agulhas Current
x_range = [int(a.x_max*0.119), int(a.x_max*0.253)]
y_range = [int(a.y_max*0.378667), int(a.y_max*0.501333)]
z_range = [0, a.z_max]
t_list = np.arange(0, 96, 24, dtype=int).tolist()
quality=-6

# the mediterranean sea region
#x_range = [int(a.x_max*0.057), int(a.x_max*0.134)]
#y_range = [int(a.y_max*0.69), int(a.y_max*0.81)]
#z_range = [0, a.z_max]
#t_list = np.arange(0, 96, 48, dtype=int).tolist()

# southern sea
#x_range = [int(a.x_max*0.4), int(a.x_max*0.47)]
#y_range = [int(a.y_max*0.33), int(a.y_max*0.45)]
#z_range = [0, a.z_max]
#t_list = np.arange(0, 8000, 2160, dtype=int).tolist()
#quality=-6

#
# scripting
#

#scriptingType = "viewer"
scriptingType = "text"
#scriptingType = ""

testing_scene="flat"
#testing_scene="sphere"

if args.scene :
    testing_scene = args.scene
if args.method :
    scriptingType = args.method

outputName_text = "text_script"
outputName_viewer = "viewer_script"

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
    bgImg = os.path.join(os.path.dirname(sys.path[0]), 'renderingApps', 'mesh', 'land.png')



if (args.save_only==False):
    # produce scripts from one of
    if (scriptingType == "viewer"):
        # 1. interactive app
        print ("launch interactive viewer")
        
        gad_dir_viewer = os.path.join(os.path.dirname(sys.path[0]), 'GAD_viewer')
        os.makedirs(gad_dir_viewer, exist_ok=True)
        # Thread(target = a.renderTask(t_list=t_list, x_range=x_range, y_range=y_range,z_range=z_range, q=quality, mode=render_mode, flip_axis=flip_axis, transpose=transpose, bgImg=bgImg, outputName=os.path.join(gad_dir_viewer, outputName_viewer))).start()
        Thread(target=a.renderTask, kwargs={
            't_list': t_list,
            'x_range': x_range,
            'y_range': y_range,
            'z_range': z_range,
            'q': quality,
            'mode': render_mode,
            'flip_axis': flip_axis,
            'transpose': transpose,
            'bgImg': bgImg,
            'outputName': os.path.join(gad_dir_viewer, outputName_viewer)
        }).start()
    

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
        gad_dir = os.path.join(os.path.dirname(sys.path[0]), 'GAD_text')
        os.makedirs(gad_dir, exist_ok=True)
        

        # Create the output directory if it doesn't exist
        
        a.generateScript(input_names, kf_interval, dims, meshType, world_bbx_len, cam, tf_range, template=script_template, outfile=os.path.join(gad_dir, outputName_text), bgImg=bgImg);
        

        #
        # download data for offline render
        #


if (args.save or args.save_only):
    output_basename = 'Out_text' if scriptingType == "text" else 'Out_viewer'
    output_dir = os.path.join(os.path.dirname(sys.path[0]), output_basename)
    print("Raw data output_dir: ", output_dir)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    a.saveRawFilesByVisusRead(t_list=t_list, x_range=x_range, y_range=y_range,z_range=z_range, q=quality, flip_axis=flip_axis, transpose=transpose, output_dir=output_dir)

