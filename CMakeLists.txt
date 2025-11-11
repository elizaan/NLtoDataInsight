cmake_minimum_required(VERSION 3.5)
project(mini_vistool)

if (NOT WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
endif()

option(BUILD_EXAMPLES "Build mini example programs" OFF)
option(BUILD_AIEXAMPLE "Build the AI example program" OFF)
option(BUILD_RENDERING_APPS "Build rendering programs" OFF)
option(BUILD_RENDERING_APPS_PY "Build rendering python libs" OFF)

add_executable(basic_usage
    basic_usage.cpp
    loader.cpp
    util.cpp)

if (BUILD_EXAMPLES)
	add_subdirectory(examples)
endif()

if (BUILD_RENDERING_APPS)
	add_subdirectory(renderingApps)
endif()

if (BUILD_RENDERING_APPS_PY)
	add_subdirectory(renderingApps/py)
endif()

if (BUILD_AIEXAMPLE)
	add_subdirectory(AIExample)
endif()
	

