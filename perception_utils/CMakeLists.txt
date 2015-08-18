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
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

rosbuild_init()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

find_package(HDF5)
#uncomment if you have defined messages
#rosbuild_genmsg()
#uncomment if you have defined services
#rosbuild_gensrv()

rosbuild_add_library(${PROJECT_NAME}
  src/perception_utils.cpp
  src/vfh/vfh_pose_estimator.cpp)

rosbuild_add_executable(vfh_train tools/vfh/vfh_train.cpp)
target_link_libraries(vfh_train ${PROJECT_NAME} ${HDF5_hdf5_LIBRARY})
target_link_libraries(vfh_train libvtkCommon.so libvtkFiltering.so libvtkRendering.so libvtkIO.so
  libvtkGraphics.so)

rosbuild_add_executable(vfh_test tools/vfh/vfh_test.cpp)
target_link_libraries(vfh_test ${PROJECT_NAME} ${HDF5_hdf5_LIBRARY})
target_link_libraries(vfh_test libvtkCommon.so libvtkFiltering.so libvtkRendering.so libvtkIO.so
  libvtkGraphics.so)

rosbuild_add_executable(vfh_generate_views tools/vfh/vfh_generate_views.cpp)
target_link_libraries(vfh_generate_views ${PROJECT_NAME} ${HDF5_hdf5_LIBRARY})
target_link_libraries(vfh_generate_views libvtkCommon.so libvtkFiltering.so libvtkRendering.so libvtkIO.so
  libvtkGraphics.so)
