cmake_minimum_required(VERSION 2.8.3)
project(sbpl_perception)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
# set(ROS_BUILD_TYPE RelWithDebInfo)
# set(ROS_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_COLOR_MAKEFILE ON)
set(ROS_COMPILE_FLAGS "-W -Wno-reorder")

# Load catkin and all dependencies required for this package
find_package(catkin REQUIRED COMPONENTS 
  deep_rgbd_utils
  cv_bridge 
  kinect_sim 
  perception_utils 
  pcl_ros 
  roscpp 
  roslib 
  sbpl 
  sbpl_utils)

find_package(Boost COMPONENTS serialization thread mpi REQUIRED)
find_package(OpenCV REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP)
# find_package(fcl REQUIRED)
find_package(octomap REQUIRED)

find_package(PkgConfig QUIET)
# pkg_check_modules(FCL REQUIRED fcl)

# TODO: why does the following line cause a nightmare?
# find_package(PCL REQUIRED)

find_path(dart_ROOT NAMES lib/libdart.so src/tracker.h HINTS
  /home/venkatrn/research/dart)
set(dart_INCLUDE_DIRS ${dart_ROOT}/src)
set(dart_LIBRARY_DIRS ${dart_ROOT}/lib)
set(dart_LIBRARIES dart)
include_directories(${dart_ROOT}/src)
link_directories(${dart_ROOT}/lib)

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIR} ${Pangolin_CONFIG_INCLUDE_DIR})

find_package(CUDA)
include_directories(${CUDA_INCLUDE_DIRS} ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)

include(CMakeForceCompiler)
CMAKE_FORCE_CXX_COMPILER(mpicxx ${CMAKE_CXX_COMPILER_ID})
# set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS} ${OpenMP_CXX_FLAGS})
# set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})

include_directories(include ${Boost_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${FCL_INCLUDE_DIRS}
  ${dart_ROOT}/src )
include_directories(MPI_INCLUDE_PATH)

catkin_package(
  CATKIN_DEPENDS 
    deep_rgbd_utils
    cv_bridge 
    kinect_sim 
    perception_utils 
    pcl_ros 
    roscpp 
    roslib 
    sbpl 
    sbpl_utils
  DEPENDS 
    # OpenCV 
    MPI 
    OpenMP 
    PCL 
    Boost
    CUDA
  INCLUDE_DIRS 
    include
    ${dart_ROOT}/src
    ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc
  LIBRARIES 
    ${PROJECT_NAME}
    ${dart_ROOT}/lib/libdart.so
    # dart
  )

add_library(${PROJECT_NAME}
  src/search_env.cpp
  src/rcnn_heuristic_factory.cpp
  src/discretization_manager.cpp
  src/graph_state.cpp
  src/object_state.cpp
  src/object_model.cpp
  src/config_parser.cpp
  src/object_recognizer.cpp
  src/utils/utils.cpp
  # src/utils/object_utils.cpp
  src/utils/dataset_generator.cpp)

target_link_libraries(${PROJECT_NAME} ${MPI_LIBRARIES} ${Boost_LIBRARIES} ${catkin_LIBRARIES}
  ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES} ${Pangolin_LIBRARIES}
  ${dart_ROOT}/lib/libdart.so
 libvtkCommon.so libvtkFiltering.so libvtkRendering.so libvtkIO.so
  # ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${FCL_LIBRARIES} libvtkCommon.so libvtkFiltering.so libvtkRendering.so libvtkIO.so
  libvtkGraphics.so)

add_executable(real_test src/experiments/experiments.cpp)
target_link_libraries(real_test ${PROJECT_NAME})

add_executable(sim_test src/experiments/sim_test.cpp)
target_link_libraries(sim_test ${PROJECT_NAME})

add_executable(demo src/experiments/demo.cpp)
target_link_libraries(demo ${PROJECT_NAME})

add_executable(ycb_demo src/experiments/ycb_demo.cpp)
target_link_libraries(ycb_demo ${PROJECT_NAME})

catkin_add_gtest(${PROJECT_NAME}_states_test tests/states_test.cpp)
target_link_libraries(${PROJECT_NAME}_states_test ${PROJECT_NAME})

catkin_add_gtest(${PROJECT_NAME}_hash_manager_test tests/hash_manager_test.cpp)
target_link_libraries(${PROJECT_NAME}_hash_manager_test ${PROJECT_NAME})


#####################################################################
# Needed only for experiments and debugging.
#####################################################################

# add_executable(heuristic_test src/experiments/heuristic_test.cpp)
# target_link_libraries(heuristic_test ${PROJECT_NAME})

# add_executable(generate_dataset src/utils/generate_dataset.cpp)
# target_link_libraries(generate_dataset ${PROJECT_NAME})

# add_executable(roi_saver
#   experiments/src/roi_saver.cpp)
# target_link_libraries(roi_saver ${PROJECT_NAME})

add_executable(ground_truth_parser
  experiments/src/ground_truth_parser.cpp)
target_link_libraries(ground_truth_parser ${PROJECT_NAME})

# add_executable(vfh_estimator
#   experiments/src/vfh_estimator.cpp)
# target_link_libraries(vfh_estimator ${PROJECT_NAME})

add_executable(perch
  experiments/src/perch.cpp)
target_link_libraries(perch ${PROJECT_NAME})

add_executable(perch_single_object
  experiments/src/perch_single_object.cpp)
target_link_libraries(perch_single_object ${PROJECT_NAME})

# add_executable(greedy_icp
#   experiments/src/greedy_icp.cpp)
# target_link_libraries(greedy_icp ${PROJECT_NAME})

# add_executable(experiments/view_generator
#   experiments/src/view_generator.cpp)
# target_link_libraries(experiments/view_generator ${PROJECT_NAME})
# target_link_libraries(experiments/view_generator libvtkCommon.so libvtkFiltering.so libvtkRendering.so libvtkIO.so
#   libvtkGraphics.so)
#
#####################################################################

# set_target_properties(vfh_estimator PROPERTIES LIBRARY_OUTPUT_DIRECTORY
#   ${CATKIN_DEVEL_PREFIX}/experiments)
# set_target_properties(perch PROPERTIES LIBRARY_OUTPUT_DIRECTORY
#   ${CATKIN_DEVEL_PREFIX}/experiments)
# set_target_properties(greedy_icp PROPERTIES LIBRARY_OUTPUT_DIRECTORY
#   ${CATKIN_DEVEL_PREFIX}/experiments)
# set_target_properties(roi_saver PROPERTIES LIBRARY_OUTPUT_DIRECTORY
#   ${CATKIN_DEVEL_PREFIX}/experiments)
# set_target_properties(ground_truth_parser PROPERTIES LIBRARY_OUTPUT_DIRECTORY
#   ${CATKIN_DEVEL_PREFIX}/experiments)
