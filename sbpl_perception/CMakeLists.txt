cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
# set(ROS_BUILD_TYPE RelWithDebInfo)
set(ROS_BUILD_TYPE Debug)
# set(ROS_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++0x")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
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
  src/discretization_manager.cpp
  src/graph_state.cpp
  src/object_state.cpp
  src/object_model.cpp
  src/search_env.cpp)
#target_link_libraries(${PROJECT_NAME} another_library)
#rosbuild_add_boost_directories()
#rosbuild_link_boost(${PROJECT_NAME} thread)

rosbuild_add_executable(experiments src/experiments/experiments.cpp)
target_link_libraries(experiments ${PROJECT_NAME})
target_link_libraries(experiments libvtkCommon.so libvtkFiltering.so libvtkRendering.so libvtkIO.so
  libvtkGraphics.so)

rosbuild_add_executable(sim_test src/experiments/sim_test.cpp)
target_link_libraries(sim_test ${PROJECT_NAME})
target_link_libraries(sim_test libvtkCommon.so libvtkFiltering.so libvtkRendering.so libvtkIO.so
  libvtkGraphics.so)

rosbuild_add_gtest(tests/states_test tests/states_test.cpp)
target_link_libraries(tests/states_test ${PROJECT_NAME})

rosbuild_add_gtest(tests/hash_manager_test tests/hash_manager_test.cpp)
target_link_libraries(tests/hash_manager_test ${PROJECT_NAME})
