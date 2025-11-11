cmake_minimum_required(VERSION 3.5)
project(vistool_examples)

if (NOT WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
endif()

find_package(OpenGL 3 REQUIRED)
find_package(glfw3 REQUIRED)
find_package(ospray REQUIRED)
find_package(rkcommon REQUIRED)

get_target_property(RKCOMMON_INCLUDE_DIRS rkcommon::rkcommon
      INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(OSPRAY_INCLUDE_DIRS ospray::ospray
      INTERFACE_INCLUDE_DIRECTORIES)

file (GLOB SOURCE_FILES
  "../osprayApps/ArcballCamera.cpp"
  "../osprayApps/TransferFunctionWidget.cpp"
  "../osprayApps/KeyframeWidget.cpp"
  "../osprayApps/clipping_plane.cpp"
  "../osprayApps/GLFWOSPWindow.cpp"
  "../osprayApps/rectMesh.cpp"
  "../imgui/*.cpp" "../../loader.cpp" "../../util.cpp" "../osprayApps/RenderFuncs.cpp")

file (GLOB PYTHON_FILES "osprayApps.cpp" "")

message(OSPRAY_INCLUDE_DIRS=${OSPRAY_INCLUDE_DIRS})
message(RKCOMMON_INCLUDE_DIRS=${RKCOMMON_INCLUDE_DIRS})

include_directories(
  PUBLIC ${OSPRAY_INCLUDE_DIRS} 
  	${RKCOMMON_INCLUDE_DIRS})

find_package(pybind11 REQUIRED)

pybind11_add_module(vistool_py
	utils.cpp 
    	../../loader.cpp
    	../../util.cpp
)
pybind11_add_module(vistool_py3
	utils3.cpp 
    	../../loader.cpp
    	../../util.cpp
)

			
target_link_libraries(vistool_py)

pybind11_add_module(vistool_py_osp
	${SOURCE_FILES}
	${PYTHON_FILES}
)

			
target_link_libraries(vistool_py_osp
			PRIVATE
			glfw
			${OPENGL_LIBRARIES}
			ospray::ospray
)

if (True)
  find_package(VTK COMPONENTS 
    CommonColor
    CommonCore
    CommonDataModel
    FiltersSources
    FiltersFlowPaths
    FiltersSources
    FiltersCore
    IOImage
    #IOLegacy
    InteractionStyle
    RenderingContextOpenGL2
    RenderingCore
    RenderingFreeType
    RenderingGL2PSOpenGL2
    RenderingOpenGL2
    RenderingVolume
    RenderingVolumeOpenGL2
    
    )

  if (NOT VTK_FOUND)
    message(FATAL_ERROR "SimpleRayCast: Unable to find the VTK build folder.")
  endif()

  pybind11_add_module(vistool_py_vtk 
    vtkApps.cpp 
    ../vtkApps/vtkFuns.cpp 
    ../../loader.cpp
    ../../util.cpp
    )

  target_link_libraries(vistool_py_vtk PRIVATE ${VTK_LIBRARIES})

  # vtk_module_autoinit is needed
  vtk_module_autoinit(
    TARGETS vistool_py_vtk
    MODULES ${VTK_LIBRARIES}
    )

  # Add improved VTK module with oceanographic visualization enhancements
  pybind11_add_module(vistool_py_vtk2 
    vtkApps2.cpp 
    ../vtkApps/vtkFuns2.cpp 
    ../../loader.cpp
    ../../util.cpp
    )

  target_link_libraries(vistool_py_vtk2 PRIVATE ${VTK_LIBRARIES})

  # vtk_module_autoinit is needed for improved module
  vtk_module_autoinit(
    TARGETS vistool_py_vtk2
    MODULES ${VTK_LIBRARIES}
    )

  pybind11_add_module(vistool_py_vtk3 
    vtkApps3.cpp 
    ../vtkApps/vtkFuns3.cpp 
    ../../loader.cpp
    ../../util.cpp
    )

  target_link_libraries(vistool_py_vtk3 PRIVATE ${VTK_LIBRARIES})

  # vtk_module_autoinit is needed for improved module
  vtk_module_autoinit(
    TARGETS vistool_py_vtk3
    MODULES ${VTK_LIBRARIES}
    )
endif()



		      

