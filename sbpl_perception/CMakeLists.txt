cmake_minimum_required(VERSION 2.8)
project(sbpl_perception)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
# set(ROS_BUILD_TYPE RelWithDebInfo)
# set(ROS_BUILD_TYPE Debug)
option(USE_CUDA "use cuda or not" ON)
if(USE_CUDA)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-8.0)
add_definitions(-DCUDA_ON)
endif()
if(USE_CUDA)
# cuda
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O3 -fopenmp --default-stream per-thread")
list(APPEND renderer_inc ${CUDA_INCLUDE_DIRS})
list(APPEND renderer_lib ${CUDA_LIBRARIES})
endif()
if(USE_CUDA)
CUDA_COMPILE(renderer_cuda_objs ${renderer_cuda})
endif()


set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_COLOR_MAKEFILE ON)
# set(ROS_COMPILE_FLAGS "-W -Wno-reorder")
#set(CMAKE_POSITION_INDEPENDENT_CODE ON)
SET(SPECIAL_OS_LIBS "pthread" "X11" "z")
find_package(CUDA REQUIRED)
set(OpenCV_DIR /usr/local/share/OpenCV)

set(CUSTOM_BOOST_MPI OFF)
message(WARNING CUSTOM_BOOST_MPI)
message(WARNING ${CUSTOM_BOOST_MPI})

include_directories(${OpenCV_INCLUDE_DIRS})
find_package(ZLIB)

include_directories(${ZLIB_INCLUDE_DIRS})
# Load catkin and all dependencies required for this package

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  kinect_sim
  cuda_renderer
  perception_utils
  pcl_ros
  roscpp
  roslib
  sbpl_perch
  eigen_conversions
  visualization_msgs
  rospy
  sbpl_utils
  sbpl_utils)


#include_directories(${CUDA_INCLUDE_DIRS})

#find_package(Boost COMPONENTS serialization thread mpi REQUIRED)

if (${CUSTOM_BOOST_MPI})
  set (BOOST_ROOT "/media/aditya/A69AFABA9AFA85D9/Cruzr/code/boost_1_67_0_perch/install")
  set (BOOSTROOT "/media/aditya/A69AFABA9AFA85D9/Cruzr/code/boost_1_67_0_perch/install")
  set (BOOST_INCLUDEDIR "/media/aditya/A69AFABA9AFA85D9/Cruzr/code/boost_1_67_0_perch/install/include")
  set (BOOST_LIBRARYDIR "/media/aditya/A69AFABA9AFA85D9/Cruzr/code/boost_1_67_0_perch/install/lib")
  set (Boost_NO_SYSTEM_PATHS ON)
  set(Boost_USE_STATIC_LIBS OFF)
  set(Boost_USE_MULTITHREADED ON)
  set(Boost_USE_STATIC_RUNTIME OFF)
  set( Boost_LIBRARIES
        ${BOOST_LIBRARYDIR}/libboost_serialization.so
        ${BOOST_LIBRARYDIR}/libboost_mpi.so)
else ()
  find_package(Boost 1.58.0 REQUIRED mpi serialization system)
endif()

include_directories(${Boost_INCLUDE_DIRS})


# find_package(OpenCV REQUIRED)
#find_package(MPI REQUIRED)

#find_package(PkgConfig REQUIRED)
#pkg_check_modules(SBPL REQUIRED sbpl_perch)
#include_directories(${SBPL_INCLUDE_DIRS})
#link_directories(${SBPL_LIBRARY_DIRS})

find_package(OpenMP)
# find_package(fcl REQUIRED)
find_package(octomap REQUIRED)

find_package(PkgConfig QUIET)
# pkg_check_modules(FCL REQUIRED fcl)

# TODO: why does the following line cause a nightmare?
# find_package(PCL REQUIRED)

#include(CMakeForceCompiler)
#CMAKE_FORCE_CXX_COMPILER(mpicxx ${CMAKE_CXX_COMPILER_ID})

#CMAKE_FORCE_CXX_COMPILER(/media/aditya/A69AFABA9AFA85D9/Cruzr/code/openmpi-4.0.0/install/bin/mpicxx ${CMAKE_CXX_COMPILER_ID})

# set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS} ${OpenMP_CXX_FLAGS})
# set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})

include_directories(include ${Boost_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${FCL_INCLUDE_DIRS})

# include_directories(${MPI_INCLUDE_PATH})
if (${CUSTOM_BOOST_MPI})
  set (MPI_INCLUDE /media/aditya/A69AFABA9AFA85D9/Cruzr/code/openmpi-4.0.0/install/include)
  set (CXX_PATH /media/aditya/A69AFABA9AFA85D9/Cruzr/code/openmpi-4.0.0/install/bin/mpicxx)
else()
  set (MPI_INCLUDE /usr/include/openmpi/)
  set (CXX_PATH /usr/bin/mpicxx)
endif()

include_directories(${MPI_INCLUDE})
#include_directories(${Pangolin_INCLUDE_DIRS})


include_directories(${catkin_INCLUDE_DIRS})

catkin_package(
  CATKIN_DEPENDS
    image_transport
    cv_bridge
    kinect_sim
    cuda_renderer
    perception_utils
    pcl_ros
    roscpp
    roslib
    sbpl_perch
    sbpl_utils
    rospy
    eigen_conversions
    visualization_msgs
  DEPENDS
    # OpenCV
    #MPI
    OpenMP
    PCL
    #Boost
  INCLUDE_DIRS
    include
    ${MPI_INCLUDE}
  LIBRARIES
    ${PROJECT_NAME}
  )

add_library(${PROJECT_NAME}
  src/rcnn_heuristic_factory.cpp
  src/discretization_manager.cpp
  src/graph_state.cpp
  src/object_state.cpp
  src/object_model.cpp
  src/search_env.cpp
  src/config_parser.cpp
  src/object_recognizer.cpp
  src/utils/utils.cpp
  # src/utils/object_utils.cpp
  src/utils/dataset_generator.cpp
  src/ColorSpace/ColorSpace.cpp
  src/ColorSpace/Comparison.cpp
  src/ColorSpace/Conversion.cpp
)

#set( MPI_CXX_LIBRARIES
#        ${MPI_ROOT}/lib/libmpi.so)
#message(WARNING ${catkin_LIBRARIES})

target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES} ${catkin_LIBRARIES} ${ZLIB_LIBRARIES}
  #${Pangolin_LIBRARIES}
  #${MPI_CXX_LIBRARIES} ${CUDA_LIBRARIES}
  #${PCL_LIBRARIES} ${OpenCV_LIBRARIES} /usr/lib/libvtkCommon.so.5.10 /usr/lib/libvtkFiltering.so.5.10 /usr/lib/libvtkRendering.so.5.10 /usr/lib/libvtkIO.so.5.10
  # ${MPI_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${FCL_LIBRARIES} libvtkCommon.so libvtkFiltering.so libvtkRendering.so libvtkIO.so
  #/usr/lib/libvtkGraphics.so.5.10
  )

#add_executable(real_test src/experiments/experiments.cpp)
#target_link_libraries(real_test ${PROJECT_NAME})

#add_executable(sim_test src/experiments/sim_test.cpp)
#target_link_libraries(sim_test ${PROJECT_NAME})



include(CMakeForceCompiler)
  #CMAKE_FORCE_CXX_COMPILER(mpicxx ${CMAKE_CXX_COMPILER_ID})

CMAKE_FORCE_CXX_COMPILER(${CXX_PATH} ${CMAKE_CXX_COMPILER_ID})

#catkin_add_gtest(${PROJECT_NAME}_states_test tests/states_test.cpp)
#target_link_libraries(${PROJECT_NAME}_states_test ${PROJECT_NAME})

#catkin_add_gtest(${PROJECT_NAME}_hash_manager_test tests/hash_manager_test.cpp)
#target_link_libraries(${PROJECT_NAME}_hash_manager_test ${PROJECT_NAME})


#####################################################################
# Needed only for experiments and debugging.
#####################################################################
set(CUDA_NVCC_FLAGS "-arch=compute_30" CACHE STRING "nvcc flags" FORCE)
set(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE) 
set(LIB_TYPE STATIC)
cuda_add_library(CUDA_getCost ${LIB_TYPE} src/compute_cost.cu)
#add_executable(roscuda_basic_node src/aaa.cpp)

#add_dependencies(roscuda_basic_node CUDA_getCost)
#TARGET_LINK_LIBRARIES(roscuda_basic_node
#${catkin_LIBRARIES}
#CUDA_getCost
#)

add_executable(demo src/experiments/demo.cpp)
add_dependencies(demo CUDA_getCost)
target_link_libraries(demo ${PROJECT_NAME} CUDA_getCost)

add_executable(heuristic_test src/experiments/heuristic_test.cpp)
add_dependencies(heuristic_test CUDA_getCost)
target_link_libraries(heuristic_test ${PROJECT_NAME} CUDA_getCost)

# add_executable(generate_dataset src/utils/generate_dataset.cpp)
# target_link_libraries(generate_dataset ${PROJECT_NAME})

# add_executable(roi_saver
#   experiments/src/roi_saver.cpp)
# target_link_libraries(roi_saver ${PROJECT_NAME})

#add_executable(ground_truth_parser
#  experiments/src/ground_truth_parser.cpp)
#target_link_libraries(ground_truth_parser ${PROJECT_NAME})

# add_executable(vfh_estimator
#   experiments/src/vfh_estimator.cpp)
# target_link_libraries(vfh_estimator ${PROJECT_NAME})

add_executable(perch
  experiments/src/perch.cpp)
add_dependencies(perch CUDA_getCost)
target_link_libraries(perch ${PROJECT_NAME} ${ZLIB_LIBRARIES} CUDA_getCost)

add_executable(perch_fat
  experiments/src/perch_fat.cpp)
add_dependencies(perch_fat CUDA_getCost)
target_link_libraries(perch_fat ${PROJECT_NAME} ${ZLIB_LIBRARIES} CUDA_getCost)

add_executable(perch_single_object
  experiments/src/perch_single_object.cpp)
add_dependencies(perch_single_object CUDA_getCost)
target_link_libraries(perch_single_object ${PROJECT_NAME} ${ZLIB_LIBRARIES} CUDA_getCost)

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

install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})

#install(DIRECTORY include/ColorSpace/
#        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  PATTERN ".svn" EXCLUDE)

install(DIRECTORY data/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/data
  PATTERN ".svn" EXCLUDE)

install(DIRECTORY visualization/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/visualization
  PATTERN ".svn" EXCLUDE)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
  PATTERN ".svn" EXCLUDE)

install(DIRECTORY demo/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/demo
  PATTERN ".svn" EXCLUDE)

install(TARGETS demo
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

#catkin_install_python(
#  PROGRAMS
#  src/scripts/render_poses.py
#  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
#)
