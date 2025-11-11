import sys, os
import numpy as np
import time
from datetime import timedelta
from threading import Thread

from OpenVisus import *
import openvisuspy as ovp
import vistool_py
import vistool_py_osp
import vistool_py_vtk2
import vtk
from vtk.util import numpy_support


# common utility functions

def getRawFileName(dimsx, dimsy, dimsz, t):
    return "ocean_{}_{}_{}_t{}_float32.raw".format(dimsx, dimsy, dimsz, int(t))
    
def getVTKFileName(dimsx, dimsy, dimsz, t):
    t = int(t)
    return "ocean_{}_{}_{}_t{}.vtk".format(dimsx, dimsy, dimsz, t)

def getRawFileNameWithPrefix(name, dimsx, dimsy, dimsz, t):
    return "{}_{}_{}_{}_t{}_float32.raw".format(name, dimsx, dimsy, dimsz, int(t))    

def saveFile(raw_fpath, data):
    start_time = time.monotonic()
    data.astype('float32').tofile(raw_fpath)
    end_time = time.monotonic()
    print('Download Duration: {}'.format(timedelta(seconds=end_time - start_time)))
    

# animation handler

class AnimationHandler:
    def __init__(self, path="", pathType="visuspy"):
        if (path != ""):
            self.srcType = pathType
            if (self.srcType == "visus"): # use visus to load
                self.dataSrc = LoadDataset(path)
                #print(self.dataSrc.getDatasetBody().toString())
            elif (pathType == "visuspy"): # set local file path
                self.dataSrc = ovp.LoadDataset(path)
            elif (pathType == "local"): # set local file path
                self.dataSrc = path

    def setDataDim(self, x_max=0, y_max=0, z_max=0, t_max=0):
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max
        self.t_max = t_max
        print("set data dims [0, {}] [0, {}] [0, {}] t=[0, {}]".format(x_max, y_max, z_max, t_max))

    # get data from source
    # TODO local data src
    def readData(self, t=0,x_range=[0,0],y_range=[0,0],z_range=[0,0],q=-6, flip_axis=2, transpose=False):
        if (self.srcType == "visus"): # use visus to read
            d = self.dataSrc.read(time=t,x=x_range,y=y_range,z=z_range,quality=q)
            
            if (flip_axis >= 0): # flip data on demand
                d = np.flip(d, flip_axis)
            if (transpose): # transpose data on demend
                d = np.transpose(d, (2, 1, 0))
            return d
        elif (self.srcType == "visuspy"): # use visus to read
            d = self.dataSrc.db.read(time=t,x=x_range,y=y_range,z=z_range,quality=q)
            if (flip_axis >= 0): # flip data on demand
                d = np.flip(d, flip_axis)
            if (transpose): # transpose data on demend
                d = np.transpose(d, (2, 1, 0))
            return d

    # get file names to save or script
    def getRawFileNames(self, dimsx, dimsy, dimsz, t_list):
        t_names = []
        counter = 0;
        for t in t_list:
            t=int(t)    
            print(t)    
            # concat all timesteps
            t_names.append(getRawFileName(dimsx, dimsy, dimsz, t))
            counter += 1
        return t_names
    
    def getVTKFileNames(self, dimsx, dimsy, dimsz, t_list):
        t_names = []
        counter = 0;
        for t in t_list:
            print(t)    
            # concat all timesteps
            t_names.append(getVTKFileName(dimsx, dimsy, dimsz, t))
            counter += 1
        return t_names
        
    def readSave(self, x_range, y_range, z_range, q, t, flip_axis, transpose, output_dir):
        print(t)
        start_time = time.monotonic()
        data = self.readData(x_range=x_range, y_range=y_range, z_range=z_range, q=q, t=t, flip_axis=flip_axis, transpose=transpose)
        end_time = time.monotonic()
        print('Read Duration: {}'.format(timedelta(seconds=end_time - start_time)))
        dims = data.shape
        # save 
        filename = getRawFileName(data.shape[2], data.shape[1], data.shape[0], int(t))
    
        # Save to the output directory
        output_path = os.path.join(output_dir, filename)
        saveFile(output_path, data)
        # saveFile(getRawFileName(data.shape[2], data.shape[1], data.shape[0], int(t)), data)

    def saveRawFilesByVisusRead(self, x_range=[0,0], y_range=[0,0], z_range=[0,0], q=-6, t_list=[0], flip_axis=2, transpose=False, output_dir = None):
        dims = [100, 100, 100]
        counter = 0;
        start_time_all = time.monotonic()
        for t in t_list:
            Thread(target = self.readSave(x_range, y_range, z_range, q, t, flip_axis, transpose, output_dir)).start()
        print("count ", counter)
        end_time_all = time.monotonic()
        print('Download Duration: {}'.format(timedelta(seconds=end_time_all - start_time_all)))
            
    def saveVTKFilesByVisusRead(self, v0, v1, v2, scalar, x_range=[0,0], y_range=[0,0], z_range=[0,0], q=-6, t_list=[0], flip_axis=2, transpose=False, output_dir = None):
        dims = [100, 100, 100]
        counter = 0;
        start_time = time.monotonic()
        #print(vtk.__version__)
        
        
        for t in t_list:
            t = int(t)
            data0=ovp.LoadDataset(v0).db.read(time=t, x=x_range,y=y_range, quality=q) 
            data1=ovp.LoadDataset(v1).db.read(time=t, x=x_range,y=y_range, quality=q) 
            data2=ovp.LoadDataset(v2).db.read(time=t, x=x_range,y=y_range, quality=q) 
            data3=ovp.LoadDataset(scalar).db.read(time=t, x=x_range,y=y_range, quality=q) 

            vx=data0
            vy=data1
            vz=data2
            s=data3
            
            dim=vx.shape
            
            # Generate the grid
            xx,yy,zz=np.mgrid[0:dim[0],0:dim[1],0:dim[2]]

            pts = np.empty(vx.shape + (3,), dtype=float)
            pts[..., 0] = xx
            pts[..., 1] = yy
            pts[..., 2] = zz
            
            vectors = np.empty(vx.shape + (3,), dtype=float)
            vectors[..., 0] = vz
            vectors[..., 1] = vy
            vectors[..., 2] = vx
            
            # We reorder the points and vectors so this is as per VTK's
            # requirement of x first, y next and z last.
            pts = pts.transpose(2, 1, 0, 3).copy()
            pts.shape = pts.size // 3, 3
            
            vectors = vectors.transpose(2, 1, 0, 3).copy()
            vectors.shape = vectors.size // 3, 3
            print(vectors.shape)
            
            s = s.transpose(2, 1, 0).copy()
            s.shape = s.size
            
            # sg = tvtk.StructuredGrid(dimensions=xx.shape, points=pts)
            sg = vtk.vtkStructuredGrid()
            sg.SetDimensions(xx.shape[0], xx.shape[1], xx.shape[2])
            points = vtk.vtkPoints()
            points.SetNumberOfPoints(pts.shape[0])

            for i in range(pts.shape[0]):
                points.SetPoint(i, pts[i, 0], pts[i, 1], pts[i, 2])
            sg.SetPoints(points)
            
            # sg.point_data.vectors = vectors
            # sg.point_data.vectors.name = 'v'
            # sg.point_data.scalars = s
            # sg.point_data.scalars.name = 's'
            # Add the vectors to the grid
            vtk_vectors = numpy_support.numpy_to_vtk(vectors, deep=1)
            vtk_vectors.SetName('v')
            sg.GetPointData().SetVectors(vtk_vectors)

            # Add the scalars to the grid
            vtk_scalars = numpy_support.numpy_to_vtk(s, deep=1)
            vtk_scalars.SetName('s')
            sg.GetPointData().SetScalars(vtk_scalars)

            fpath = getVTKFileName(data0.shape[2], data0.shape[1], data0.shape[0], int(t))

            if output_dir:
                fpath = os.path.join(output_dir, fpath)
            
            # write_data(sg, fpath)
            writer = vtk.vtkStructuredGridWriter()
            writer.SetFileName(fpath)
            writer.SetInputData(sg)
            writer.Write()

            counter += 1  
        print("count ", counter)
        end_time = time.monotonic()
        print('Download Duration: {}'.format(timedelta(seconds=end_time - start_time)))
            
    # generate scripts by templates
    def generateScript(self, input_names, kf_interval, dims, meshType, world_bbx_len, cam, tf_range, template="fixedCam", s=0, e=0, dist=0, outfile="script", bgImg=""):
        if (template == "fixedCam"):
            # print("generating fixed camera script to: ", outfile, "\n")
            # convert to strict data types
            dims = np.array(dims)
            cam = np.float32(cam)
            tf_range = np.float32(tf_range)
            vistool_py.generateScriptFixedCam(outfile, input_names, kf_interval, dims, meshType, world_bbx_len, cam, tf_range, bgImg);
        elif (template == "rotate"):
            # print("generating rotating camera script to: ", outfile, "\n")
            # convert to strict data types
            dims = np.array(dims)
            tf_range = np.float32(tf_range)
            vistool_py.generateScriptRotate(outfile, input_names, kf_interval, dims, meshType, world_bbx_len, s, e, dist, tf_range, bgImg);
            
    def generateScriptStreamline(self, input_names, kf_interval, dims, meshType, world_bbx_len, cam, tf_range, tf_colors, tf_opacities, scalar_field, template="fixedCam", outfile="script", bgImg=""):
        if (template == "fixedCam"):
            # print("generating fixed camera streamline script to: ", outfile, "\n")
            # convert to strict data types
            dims = np.array(dims)
            cam = np.float32(cam)
            tf_range = np.float32(tf_range)
            tf_colors = np.float32(tf_colors)
            tf_opacities = np.float32(tf_opacities)
            vistool_py.generateScriptFixedCamStreamline(outfile, input_names, kf_interval, dims, meshType, world_bbx_len, cam, tf_range, tf_colors, tf_opacities, scalar_field, bgImg);

    # read scripts by file path
    def readScript(self, p):
        return vistool_py_osp.readScript(p);
    
    # launch rendering
    def renderTask(self, x_range=[0,0], y_range=[0,0], z_range=[0,0], q=-6, t_list=[0], flip_axis=2, transpose=False, mode=0, bgImg="", outputName="viewer_script"):
        dims = [100, 100, 100]
        total_data = []
        t_names = []
        counter = 0;
        print("generating script to: ", outputName, "\n")
        for t in t_list:
            print(t)
            start_time = time.monotonic()
            data = self.readData(t=t, x_range=x_range, y_range=y_range, z_range=z_range, q=q, flip_axis=flip_axis, transpose=transpose)
            end_time = time.monotonic()
            print('Read Duration: {}'.format(timedelta(seconds=end_time - start_time)))
            
            # concate all timesteps
            dims = data.shape
            total_data = np.concatenate((total_data, data.ravel()), axis=None)
            t_names.append(getRawFileName(data.shape[2], data.shape[1], data.shape[0], t))
            counter += 1

            # save a copy of the data if needed
            if (0):
                saveFile(getRawFileName(data.shape[2], data.shape[1], data.shape[0], t))
                
        print(dims)
        print("count ", counter)
        print(t_names)
        #print(total_data.shape)

        vistool_py_osp.init_app(sys.argv)
        vistool_py_osp.run_app(total_data, t_names, dims[2], dims[1], dims[0], counter, mode, bgImg, outputName)
        #return dims


    # launch rendering
    def renderTaskOffline(self, jsonStr):
        vistool_py_osp.init_app(sys.argv)

        output_dir = os.environ.get("RENDER_OUTPUT_DIR", "")
        if output_dir:
            print(f"Using output directory: {output_dir}")

        # vistool_py_osp.run_offline_app(jsonStr, "", -2)
        vistool_py_osp.run_offline_app(jsonStr, output_dir, -2)

    # launch vtk rendering
    def renderTaskOfflineVTK(self, jsonStr):
        output_dir = os.environ.get("RENDER_OUTPUT_DIR", "")
        if output_dir:
            print(f"Using output directory for VTK rendering: {output_dir}")
        vistool_py_vtk2.run_offline_app_improved(jsonStr, output_dir, -2)
        # vistool_py_vtk.run_offline_app(jsonStr, output_dir, -2)

    
