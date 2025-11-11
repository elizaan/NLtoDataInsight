# vis_user_tool

## animation system for large scivis data
json based textfiles + loader for automated animation scripting and production

**build**

dependencies includes:
ospray v2.12 (for rendering)
pybind11 (for python interface)

To initiate virtual environment:
```
source venv_new/bin/activate
```

To install python package dependencies:

```
pip install -r requirements.txt
```
build and install ospray following https://github.com/RenderKit/ospray/tree/release-2.12.x

set ospray_DIR and rkcommon_DIR (included in ospray build&install)

if version mismatch between conda environment vs C++ standard library
```
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python python/scripting.py -save
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python python/render.py text_script.json
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python python/scripting.py -save -method viewer -scene sphere
```

**examples/**

examples with opengl and ospray v2.12

**python/**

python interface with examples. Please run all in vis_user_tool folder:

```
// to read, download and script by either preset templates, for saving raw files and generating text jsons
// produce a list of json files
python3 python/scripting.py -save
// for help
python3 python/scripting.py -h 

// for interactive viewer
python3 python/scripting.py -save -method viewer -scene sphere/flat 
// render with json file
python3 python/render.py path_to_js
```
**jupter_notebook_example/**

remote data access through jupternotebook, see animationToolTutorial.ipynb.

**renderingApps/**

different rendering backends with raw data 

**MLLM-Aided Interface**
how to run:
```
python AIExample/script/openai_run.py
```
**Case Studies**
```
python3 python/case_study_1-1/script_cs1-1.py -save
python python/render.py python/case_study_1-1/GAD_text/case1_script.json

python python/case_study_1-3/script_cs1-3.py -save
python python/render.py python/case_study_1-3/GAD_text/case1_script.json

```
## Images

![This is an alt text.](/interactApp/demo_img/render_full_res_overview.png "This is a sample image.")
![This is an alt text.](/interactApp/demo_img/render_full_res_local.png "This is a sample image.")

## Test vtk rendering with new land-ocean volumes and colorings
"venv_new/bin/python -m pip freeze"
0. initiate virtulal anvironent as described above, install python packages using requirements.txt
1. run superbuild.sh
2. to change vtk codes in vtk files go to renderingApps->vtkApps->vtkFuns2.cpp
3. cd build, make -j
4. check in python/renderInterface.py in line 288 has vistool_py_vtk2
5. to download vtk files and json files: python python/test/test-vtk.py -save
6. run python python/renderVTK.py python/test/GAD_text/case2_script.json to render with vtk
7. file gets saved in here: python/test/rendered_frames/improved_.......