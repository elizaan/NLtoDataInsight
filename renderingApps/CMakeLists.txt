option(BUILD_OSPRAY_APPS "Build ospray rendering programs" OFF)
option(BUILD_VTK_APPS "Build vtk rendering programs" OFF)

if (BUILD_OSPRAY_APPS)
	add_subdirectory(osprayApps)
endif()

if (BUILD_VTK_APPS)
	add_subdirectory(vtkApps)
endif()
