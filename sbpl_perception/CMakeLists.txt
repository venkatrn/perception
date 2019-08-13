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
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_COLOR_MAKEFILE ON)
# set(ROS_COMPILE_FLAGS "-W -Wno-reorder")
#set(CMAKE_POSITION_INDEPENDENT_CODE ON)
SET(SPECIAL_OS_LIBS "pthread" "X11" "z")


find_package(ZLIB)

include_directories(${ZLIB_INCLUDE_DIRS})
# Load catkin and all dependencies required for this package
#find_package(CUDA REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  kinect_sim
  perception_utils
  pcl_ros
  roscpp
  roslib
  sbpl
  eigen_conversions
  visualization_msgs
  rospy
  sbpl_utils
  sbpl_utils)


#include_directories(${CUDA_INCLUDE_DIRS})

#find_package(Boost COMPONENTS serialization thread mpi REQUIRED)

#set (BOOST_ROOT "/media/aditya/A69AFABA9AFA85D9/Cruzr/code/boost_1_67_0_perch/install")
#set (BOOSTROOT "/media/aditya/A69AFABA9AFA85D9/Cruzr/code/boost_1_67_0_perch/install")
#set (BOOST_INCLUDEDIR "/media/aditya/A69AFABA9AFA85D9/Cruzr/code/boost_1_67_0_perch/install/include")
#set (BOOST_LIBRARYDIR "/media/aditya/A69AFABA9AFA85D9/Cruzr/code/boost_1_67_0_perch/install/lib")
#set (Boost_NO_SYSTEM_PATHS ON)
#set(Boost_USE_STATIC_LIBS OFF)
#set(Boost_USE_MULTITHREADED ON)
#set(Boost_USE_STATIC_RUNTIME OFF)
include_directories(${Boost_INCLUDE_DIRS})
#set( Boost_LIBRARIES
#        ${BOOST_LIBRARYDIR}/libboost_serialization.so
#        ${BOOST_LIBRARYDIR}/libboost_mpi.so)

find_package(Boost 1.58.0 REQUIRED mpi serialization system)

# find_package(OpenCV REQUIRED)
#find_package(MPI REQUIRED)

#find_package(PkgConfig REQUIRED)
#pkg_check_modules(SBPL REQUIRED sbpl)
#include_directories(${SBPL_INCLUDE_DIRS})
#link_directories(${SBPL_LIBRARY_DIRS})

find_package(OpenMP)
# find_package(fcl REQUIRED)
find_package(octomap REQUIRED)
# find_package(Pangolin REQUIRED)

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
# set (MPI_ROOT /media/aditya/A69AFABA9AFA85D9/Cruzr/code/openmpi-4.0.0/install)
set (MPI_ROOT /usr/include/openmpi/)
# include_directories(${MPI_ROOT}/include)
include_directories(${MPI_ROOT})


include_directories(${catkin_INCLUDE_DIRS})

catkin_package(
  CATKIN_DEPENDS
    cv_bridge
    kinect_sim
    perception_utils
    pcl_ros
    roscpp
    roslib
    sbpl
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
    # ${MPI_ROOT}/include/
    ${MPI_ROOT}/
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

add_executable(demo src/experiments/demo.cpp)
target_link_libraries(demo ${PROJECT_NAME})

  include(CMakeForceCompiler)
  #CMAKE_FORCE_CXX_COMPILER(mpicxx ${CMAKE_CXX_COMPILER_ID})

  # CMAKE_FORCE_CXX_COMPILER(/media/aditya/A69AFABA9AFA85D9/Cruzr/code/openmpi-4.0.0/install/bin/mpicxx ${CMAKE_CXX_COMPILER_ID})
  CMAKE_FORCE_CXX_COMPILER(/usr/bin/mpicxx ${CMAKE_CXX_COMPILER_ID})

#catkin_add_gtest(${PROJECT_NAME}_states_test tests/states_test.cpp)
#target_link_libraries(${PROJECT_NAME}_states_test ${PROJECT_NAME})

#catkin_add_gtest(${PROJECT_NAME}_hash_manager_test tests/hash_manager_test.cpp)
#target_link_libraries(${PROJECT_NAME}_hash_manager_test ${PROJECT_NAME})


#####################################################################
# Needed only for experiments and debugging.
#####################################################################

add_executable(heuristic_test src/experiments/heuristic_test.cpp)
target_link_libraries(heuristic_test ${PROJECT_NAME})

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
target_link_libraries(perch ${PROJECT_NAME})

add_executable(perch_fat
  experiments/src/perch_fat.cpp)
target_link_libraries(perch_fat ${PROJECT_NAME})

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

catkin_install_python(
  PROGRAMS
  src/scripts/render_poses.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
