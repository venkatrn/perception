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
set(ROS_COMPILE_FLAGS "-W -Wno-reorder")


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

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
include(CMakeForceCompiler)
CMAKE_FORCE_CXX_COMPILER(mpicxx ${CMAKE_CXX_COMPILER_ID})
# set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS} ${OpenMP_CXX_FLAGS})
# set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
include_directories(MPI_INCLUDE_PATH)

#uncomment if you have defined messages
#rosbuild_genmsg()
#uncomment if you have defined services
#rosbuild_gensrv()

#common commands for building c++ executables and libraries
rosbuild_add_boost_directories()
rosbuild_add_library(${PROJECT_NAME}
  src/rcnn_heuristic_factory.cpp
  src/discretization_manager.cpp
  src/graph_state.cpp
  src/object_state.cpp
  src/object_model.cpp
  src/search_env.cpp
  src/config_parser.cpp
  src/object_recognizer.cpp
  src/utils/utils.cpp
  src/utils/dataset_generator.cpp)
#target_link_libraries(${PROJECT_NAME} another_library)
#rosbuild_link_boost(${PROJECT_NAME} thread)
rosbuild_link_boost(${PROJECT_NAME} mpi serialization)

rosbuild_add_executable(real_test src/experiments/experiments.cpp)
target_link_libraries(real_test ${PROJECT_NAME})
target_link_libraries(real_test libvtkCommon.so libvtkFiltering.so libvtkRendering.so libvtkIO.so
  libvtkGraphics.so)
rosbuild_link_boost(real_test mpi serialization)
rosbuild_add_openmp_flags(real_test)

rosbuild_add_executable(heuristic_test src/experiments/heuristic_test.cpp)
target_link_libraries(heuristic_test ${PROJECT_NAME})
target_link_libraries(heuristic_test libvtkCommon.so libvtkFiltering.so libvtkRendering.so libvtkIO.so
  libvtkGraphics.so)
rosbuild_link_boost(heuristic_test mpi serialization)
rosbuild_add_openmp_flags(heuristic_test)

rosbuild_add_executable(sim_test src/experiments/sim_test.cpp)
target_link_libraries(sim_test ${PROJECT_NAME})
target_link_libraries(sim_test libvtkCommon.so libvtkFiltering.so libvtkRendering.so libvtkIO.so
  libvtkGraphics.so)
rosbuild_link_boost(sim_test mpi serialization)
rosbuild_add_openmp_flags(sim_test)

rosbuild_add_executable(generate_dataset src/utils/generate_dataset.cpp)
target_link_libraries(generate_dataset ${PROJECT_NAME})

# rosbuild_add_executable(mpi_test src/experiments/search_mpi_test.cpp)
# rosbuild_link_boost(mpi_test mpi serialization)
# target_link_libraries(mpi_test ${PROJECT_NAME})
#
# rosbuild_add_gtest(tests/states_test tests/states_test.cpp)
# target_link_libraries(tests/states_test ${PROJECT_NAME})
#
# rosbuild_add_gtest(tests/hash_manager_test tests/hash_manager_test.cpp)
# target_link_libraries(tests/hash_manager_test ${PROJECT_NAME})
#
# rosbuild_add_executable(experiments/ground_truth_parser
#   experiments/src/ground_truth_parser.cpp)
# target_link_libraries(experiments/ground_truth_parser ${PROJECT_NAME})
#
# rosbuild_add_executable(experiments/view_generator
#   experiments/src/view_generator.cpp)
# target_link_libraries(experiments/view_generator ${PROJECT_NAME})
# target_link_libraries(experiments/view_generator libvtkCommon.so libvtkFiltering.so libvtkRendering.so libvtkIO.so
#   libvtkGraphics.so)
#
# rosbuild_add_executable(experiments/vfh_estimator
#   experiments/src/vfh_estimator.cpp)
# target_link_libraries(experiments/vfh_estimator ${PROJECT_NAME})
#
# rosbuild_add_executable(experiments/perch
#   experiments/src/perch.cpp)
# target_link_libraries(experiments/perch ${PROJECT_NAME})
#
# rosbuild_add_executable(experiments/greedy_icp
#   experiments/src/greedy_icp.cpp)
# target_link_libraries(experiments/greedy_icp ${PROJECT_NAME})
# rosbuild_add_openmp_flags(experiments/greedy_icp)
