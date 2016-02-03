cmake_minimum_required(VERSION 2.8.3)
project(object_recognition_node)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)
set(ROS_BUILD_TYPE Release)
# set(ROS_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++0x")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(catkin REQUIRED COMPONENTS roscpp sensor_msgs perception_utils
  keyboard tf tf_conversions)
find_package(Boost REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include ${catkin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})

add_library(${PROJECT_NAME}
  src/perception_interface.cpp)
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${PCL_LIBRARIES})

catkin_package(
    CATKIN_DEPENDS roscpp sensor_msgs perception_utils keyboard tf tf_conversions
    DEPENDS
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
)

# Ugh, change package name so we can use object_recognition_node as target
# name.
add_executable(obj_rec_node src/object_recognition_node.cpp)
target_link_libraries(obj_rec_node ${PROJECT_NAME} libvtkCommon.so libvtkFiltering.so libvtkRendering.so libvtkIO.so
  libvtkGraphics.so)
