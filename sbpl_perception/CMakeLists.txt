cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)
set(ROS_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-std=c++0x")
# set(CMAKE_COLOR_MAKEFILE ON)


rosbuild_init()
#find_package(PCL 1.7 REQUIRED)
#include_directories(BEFORE ${PCL_INCLUDE_DIRS})
#link_directories(${PCL_LIBRARY_DIRS})
#add_definitions(${PCL_DEFINITIONS})
#if (NOT PCL_FOUND)
#  MESSAGE(FATAL_ERROR "PCL not found.\n")
#endif (NOT PCL_FOUND)


find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} )

find_package(HDF5)

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#uncomment if you have defined messages
#rosbuild_genmsg()
#uncomment if you have defined services
#rosbuild_gensrv()

#common commands for building c++ executables and libraries
rosbuild_add_library(${PROJECT_NAME}
  src/vfh_pose_estimation.cpp
  src/object_model.cpp
  src/perception_interface.cpp
  src/perception_utils.cpp
  src/search_env.cpp)
#target_link_libraries(${PROJECT_NAME} another_library)
#rosbuild_add_boost_directories()
#rosbuild_link_boost(${PROJECT_NAME} thread)

rosbuild_add_executable(object_recognition_node src/object_recognition_node.cpp)
target_link_libraries(object_recognition_node ${PROJECT_NAME})
target_link_libraries(object_recognition_node libvtkCommon.so libvtkFiltering.so libvtkRendering.so libvtkIO.so
  libvtkGraphics.so)

rosbuild_add_executable(sim_test src/experiments/sim_test.cpp)
target_link_libraries(sim_test ${PROJECT_NAME})
target_link_libraries(sim_test libvtkCommon.so libvtkFiltering.so libvtkRendering.so libvtkIO.so
  libvtkGraphics.so)

rosbuild_add_executable(experiments src/experiments/experiments.cpp)
target_link_libraries(experiments ${PROJECT_NAME})
target_link_libraries(experiments libvtkCommon.so libvtkFiltering.so libvtkRendering.so libvtkIO.so
  libvtkGraphics.so)

rosbuild_add_executable(vfh_train src/vfh_train.cpp)
target_link_libraries(vfh_train ${PROJECT_NAME} ${HDF5_hdf5_LIBRARY})
target_link_libraries(vfh_train libvtkCommon.so libvtkFiltering.so libvtkRendering.so libvtkIO.so
  libvtkGraphics.so)

rosbuild_add_executable(vfh_test src/vfh_test.cpp)
target_link_libraries(vfh_test ${PROJECT_NAME} ${HDF5_hdf5_LIBRARY})
target_link_libraries(vfh_test libvtkCommon.so libvtkFiltering.so libvtkRendering.so libvtkIO.so
  libvtkGraphics.so)

rosbuild_add_executable(vfh_generate_views src/vfh_generate_views.cpp)
target_link_libraries(vfh_generate_views ${PROJECT_NAME} ${HDF5_hdf5_LIBRARY})
target_link_libraries(vfh_generate_views libvtkCommon.so libvtkFiltering.so libvtkRendering.so libvtkIO.so
  libvtkGraphics.so)
# rosbuild_add_executable(search_env src/search_env.cpp)
# target_link_libraries(search_env ${PROJECT_NAME})
# target_link_libraries(search_env libvtkCommon.so libvtkFiltering.so libvtkRendering.so libvtkIO.so
#   libvtkGraphics.so)
