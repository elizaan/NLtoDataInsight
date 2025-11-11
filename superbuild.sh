#!/bin/bash

mkdir dependency_dir
cd dependency_dir/
git clone https://github.com/RenderKit/ospray.git
cd ospray
git checkout v2.12.0
mkdir build
cd build
# may need for first run
sudo apt-get install cmake-curses-gui
sudo apt-get install libtbb-dev
sudo apt-get install libxinerama-dev
sudo apt-get install libxcursor-dev
sudo apt-get install libxi-devlibx
sudo apt-get install pybind11-dev
sudo apt-get install xorg-dev
sudo apt-get install libglew-dev
sudo apt-get install libglfw3-dev

cmake ../scripts/superbuild
cmake --build .
dp_path=$(pwd)

# build vtk
cd ../../

git clone --recursive https://gitlab.kitware.com/vtk/vtk.git 
git checkout v9.3.1
cd vtk
mkdir build
cd build

cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX:PATH=$(pwd)/install ..
install_path=$(pwd)/install/lib/cmake/vtk-9.4
make -j
make install

echo $dp_path
cd ../../../
mkdir build
cd build
echo cmake -Dospray_DIR=$dp_path/install/ospray/lib/cmake/ospray-2.12.0 -Drkcommon_DIR=$dp_path/install/rkcommon/lib/cmake/rkcommon-1.11.0 -DVTK_DIR=$install_path -DBUILD_RENDERING_APPS_PY=ON ..
cmake -Dospray_DIR=$dp_path/install/ospray/lib/cmake/ospray-2.12.0 -Drkcommon_DIR=$dp_path/install/rkcommon/lib/cmake/rkcommon-1.11.0 -DVTK_DIR=$install_path -DBUILD_RENDERING_APPS_PY=ON -DBUILD_RENDERING_APPS=OFF -DBUILD_VTK_APPS=ON .. 
make -j

# cmake -Dospray_DIR=/home/eliza89/PhD/codes/vis_user_tool/dependency_dir/ospray/build/install/ospray/lib/cmake/ospray-2.12.0 -Drkcommon_DIR=/home/eliza89/PhD/codes/vis_user_tool/dependency_dir/ospray/build/install/rkcommon/lib/cmake/rkcommon-1.11.0 -DVTK_DIR=/home/eliza89/PhD/codes/vis_user_tool/dependency_dir/vtk/build/install/lib/cmake/vtk-9.4 -DBUILD_RENDERING_APPS_PY=ON ..
# printf "file '%s'\n" img_*_f*.png | sort -V > list.txt
# ffmpeg -f concat -safe 0 -r 4 -i list.txt -c:v rawvideo -pix_fmt yuv420p output_uncompressed.avi

# ffmpeg -f concat -safe 0 -r 4 -i list.txt -c:v libx264 -preset ultrafast -crf 0 -pix_fmt yuv420p output_lossless.mp4
# better
# ffmpeg -f concat -safe 0 -r 4 -i list.txt -c:v libx264 -preset ultrafast -crf 0 -pix_fmt yuv444p output_lossless.mp4

# # verify
# ffprobe -v error -show_entries stream=codec_name,profile output_lossless.mp4
