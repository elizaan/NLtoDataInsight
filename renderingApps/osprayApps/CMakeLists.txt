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
      
add_executable(interactiveRender
    RenderFuncs.cpp
    interactiveRenderMain.cpp
    ArcballCamera.cpp
    clipping_plane.cpp
    TransferFunctionWidget.cpp
    KeyframeWidget.cpp 
    GLFWOSPWindow.cpp
    ../imgui/imgui.cpp
    ../imgui/imgui_draw.cpp
    ../imgui/imgui_tables.cpp
    ../imgui/imgui_widgets.cpp
    ../imgui/imgui_impl_glfw_gl3.cpp
    ../../loader.cpp
    ../../util.cpp
    rectMesh.cpp)
 
message(OSPRAY_INCLUDE_DIRS=${OSPRAY_INCLUDE_DIRS})
message(RKCOMMON_INCLUDE_DIRS=${RKCOMMON_INCLUDE_DIRS})

target_include_directories(interactiveRender
			PUBLIC
			 ${OSPRAY_INCLUDE_DIRS}
			 ${RKCOMMON_INCLUDE_DIRS})
			
target_link_libraries(interactiveRender 
			glfw
			${OPENGL_LIBRARIES}
			ospray::ospray)



add_executable(offlineRender
    RenderFuncs.cpp
    offlineRenderMain.cpp
    ArcballCamera.cpp
    clipping_plane.cpp
    TransferFunctionWidget.cpp
    KeyframeWidget.cpp
    GLFWOSPWindow.cpp
    ../imgui/imgui.cpp
    ../imgui/imgui_draw.cpp
    ../imgui/imgui_tables.cpp
    ../imgui/imgui_widgets.cpp
    ../imgui/imgui_impl_glfw_gl3.cpp
    ../../loader.cpp
    ../../util.cpp
    rectMesh.cpp)

target_include_directories(offlineRender
			PUBLIC
			${OSPRAY_INCLUDE_DIRS} 
			${RKCOMMON_INCLUDE_DIRS})
			
target_link_libraries(offlineRender 
			glfw
			${OPENGL_LIBRARIES}
			ospray::ospray)





    

