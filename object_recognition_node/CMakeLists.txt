cmake_minimum_required(VERSION 2.8.3)
project(object_recognition_node)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CMakeForceCompiler)
CMAKE_FORCE_CXX_COMPILER(mpicxx ${CMAKE_CXX_COMPILER_ID})

find_package(catkin REQUIRED genmsg actionlib_msgs actionlib 
  COMPONENTS roscpp sensor_msgs perception_utils
  keyboard tf tf_conversions eigen_conversions sbpl_perception)
find_package(Boost COMPONENTS mpi REQUIRED)
find_package(MPI REQUIRED)

# find_package(catkin REQUIRED genmsg actionlib_msgs actionlib)
add_action_files(DIRECTORY action FILES DoPerch.action)
add_service_files(FILES LocalizeObjects.srv)

include_directories(${PROJECT_SOURCE_DIR}/include ${catkin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
include_directories(MPI_INCLUDE_PATH)

generate_messages(
  DEPENDENCIES
  actionlib_msgs
  std_msgs
  geometry_msgs
  sensor_msgs
  tf
  )

catkin_package(
  CATKIN_DEPENDS roscpp sensor_msgs perception_utils keyboard tf tf_conversions
  sbpl_perception
  DEPENDS MPI Boost
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  )

add_library(${PROJECT_NAME}
  src/perception_interface.cpp)
target_link_libraries(${PROJECT_NAME} ${MPI_LIBRARIES} ${catkin_LIBRARIES} ${PCL_LIBRARIES}
  ${Boost_LIBRARIES})

install(PROGRAMS
  scripts/pr2_perch_client.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

# Ugh, change package name so we can use object_recognition_node as target
# name.
add_executable(obj_rec_node src/object_recognition_node.cpp)
target_link_libraries(obj_rec_node ${PROJECT_NAME} libvtkCommon.so libvtkFiltering.so libvtkRendering.so libvtkIO.so
  libvtkGraphics.so)

add_executable(object_localizer src/object_localizer_service.cpp)
target_link_libraries(object_localizer ${catkin_LIBRARIES} ${PROJECT_NAME} libvtkCommon.so libvtkFiltering.so libvtkRendering.so libvtkIO.so
  libvtkGraphics.so)

add_executable(object_localizer_client_example
  src/test/object_localizer_client_example.cpp)
target_link_libraries(object_localizer_client_example ${catkin_LIBRARIES} ${PROJECT_NAME} libvtkCommon.so libvtkFiltering.so libvtkRendering.so libvtkIO.so
  libvtkGraphics.so)
