cmake_minimum_required(VERSION 2.8.3)
project(perception_utils)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)
# set(ROS_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(catkin REQUIRED COMPONENTS pcl_ros roscpp pcl_conversions
  cv_bridge)
find_package(Boost REQUIRED COMPONENTS filesystem program_options)
#set(OpenCV_DIR /usr/local/share/OpenCV)
find_package(OpenCV REQUIRED)
find_package(HDF5 REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS})

catkin_package(
  DEPENDS
    PCL
    Boost
    HDF5
    OpenCV
  CATKIN_DEPENDS
    pcl_ros
    roscpp
    pcl_conversions
    cv_bridge
  INCLUDE_DIRS
    include
  LIBRARIES
    ${PROJECT_NAME}
  )

add_library(${PROJECT_NAME}
  src/perception_utils.cpp
  src/vfh/vfh_pose_estimator.cpp)
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${HDF5_hdf5_LIBRARY}
  ${PCL_LIBRARIES} ${OpenCV_LIBS}
  #/usr/lib/libvtkCommon.so.5.10 /usr/lib/libvtkFiltering.so.5.10 /usr/lib/libvtkRendering.so.5.10 /usr/lib/libvtkIO.so.5.10
  )


#add_executable(vfh_train tools/vfh/vfh_train.cpp)
#target_link_libraries(vfh_train ${PROJECT_NAME})

#add_executable(vfh_test tools/vfh/vfh_test.cpp)
#target_link_libraries(vfh_test ${PROJECT_NAME})

#add_executable(vfh_generate_views tools/vfh/vfh_generate_views.cpp)
#target_link_libraries(vfh_generate_views ${PROJECT_NAME})

#add_executable(align_pcd_to_world tools/align_pcd_to_world.cpp)
#target_link_libraries(align_pcd_to_world ${PROJECT_NAME})

#add_executable(depth_image_smoother
#  tools/depth_image_smoother.cpp)
#target_link_libraries(depth_image_smoother ${PROJECT_NAME})

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
