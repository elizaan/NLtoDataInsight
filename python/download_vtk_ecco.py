import numpy as np
import pyvista as pv
from matplotlib.colors import ListedColormap
from OpenVisus import *

# path to the rendering app libs
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'build', 'renderingApps', 'py'))
import renderInterface

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

print(data.shape, d_max, d_min)




# set script details
input_names = a.getVTKFileNames(data.shape[2], data.shape[1], data.shape[0], times)
kf_interval = 1 # frames per keyframe
dims = [data.shape[2], data.shape[1], data.shape[0]] # flip axis from py array
meshType = "streamline"
world_bbx_len = 10
cam = [4.735267639160156, -3.0468220710754395, -6.374855995178223, 0, 0.6900219321250916, 0.7233046889305115, 0, 0.7236120104789734, -0.6902016997337341] # camera params, pos, dir, up
tf_range = [d_min, d_max]
meshType = "structured"

# generate script
a.generateScriptStreamline(input_names, kf_interval, dims, meshType, world_bbx_len, cam, tf_range, template="fixedCam", outfile="case2_script");


