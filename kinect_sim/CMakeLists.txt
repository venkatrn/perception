# Catkin User Guide: http://www.ros.org/doc/groovy/api/catkin/html/user_guide/user_guide.html
# Catkin CMake Standard: http://www.ros.org/doc/groovy/api/catkin/html/user_guide/standards.html
cmake_minimum_required(VERSION 2.8.3)
project(kinect_sim)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)
set(ROS_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Boost REQUIRED)
# find_package(OpenCV REQUIRED)
find_package(catkin REQUIRED COMPONENTS roscpp roslib pcl_ros cv_bridge)

include_directories(${PROJECT_SOURCE_DIR}/include ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

# Try to find GLEW library and include path.
# Once done this will define
#
# GLEW_FOUND
# GLEW_INCLUDE_PATH
# GLEW_LIBRARY

IF (WIN32)
FIND_PATH( GLEW_INCLUDE_PATH GL/glew.h
           $ENV{PROGRAMFILES}/GLEW/include
           ${PROJECT_SOURCE_DIR}/src/nvgl/glew/include
           DOC "The directory where GL/glew.h resides")
FIND_LIBRARY( GLEW_LIBRARY
              NAMES glew GLEW glew32 glew32s
              PATHS
              $ENV{PROGRAMFILES}/GLEW/lib
              ${PROJECT_SOURCE_DIR}/src/nvgl/glew/bin
              ${PROJECT_SOURCE_DIR}/src/nvgl/glew/lib
              DOC "The GLEW library")
ELSE (WIN32)
FIND_PATH( GLEW_INCLUDE_PATH GL/glew.h
           /usr/include
           /usr/local/include
           /sw/include
           /opt/local/include
           DOC "The directory where GL/glew.h resides")
FIND_LIBRARY( GLEW_LIBRARY
              NAMES GLEW glew
              PATHS
              /usr/lib64
              /usr/lib
              /usr/local/lib64
              /usr/local/lib
              /sw/lib
              /opt/local/lib
              DOC "The GLEW library")
ENDIF (WIN32)

IF (GLEW_INCLUDE_PATH)
SET( GLEW_FOUND 1 CACHE STRING "Set to 1 if GLEW is found, 0 otherwise")
ELSE (GLEW_INCLUDE_PATH)
SET( GLEW_FOUND 0 CACHE STRING "Set to 1 if GLEW is found, 0 otherwise")
ENDIF (GLEW_INCLUDE_PATH)

MARK_AS_ADVANCED( GLEW_FOUND )

FIND_PACKAGE(GLUT REQUIRED)
## Find required dependencies
FIND_PACKAGE(OpenGL REQUIRED QUIET)
#FIND_PACKAGE(GLEW REQUIRED)
## Apply required dependencies settings
LIST(APPEND LINK_LIBS "${OPENGL_LIBRARIES};${GLEW_LIBRARY};${GLUT_LIBRARY};${OPENGL_LIBRARIES}"
  )
LIST(APPEND LIB_DIRS  "${OPENGL_LIBRARY_DIR};${GLEW_LIBRARY_DIR}" )
LINK_DIRECTORIES(${LIB_DIRS})
LINK_LIBRARIES(${LINK_LIBS})
MESSAGE(STATUS "link dirs: ${LIB_DIRS}")
MESSAGE(STATUS "link libs: ${LINK_LIBS}")

catkin_package(
    CATKIN_DEPENDS 
      roscpp 
      roslib 
      pcl_ros 
      cv_bridge
    DEPENDS 
      PCL 
      Eigen
    INCLUDE_DIRS 
      include
    LIBRARIES 
      ${PROJECT_NAME}
)

add_library(${PROJECT_NAME}
  src/simulation_io.cpp
  src/camera.cpp
  src/glsl_shader.cpp
  src/model.cpp
  src/range_likelihood.cpp
  src/scene.cpp
  src/sum_reduce.cpp)

target_link_libraries (${PROJECT_NAME} ${Boost_LIBRARIES} ${catkin_LIBRARIES}
                       ${VTK_IO_TARGET_LINK_LIBRARIES}
                       ${GLEW_LIBRARIES} ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES}
                       ${GLEW_LIBRARIES} libvtkCommon.so libvtkFiltering.so
                       libvtkRendering.so libvtkIO.so)

add_executable(kinect_sim_viewer tools/sim_viewer.cpp)
add_executable(kinect_sim_test_simple tools/sim_test_simple.cpp)
add_executable(kinect_sim_test_performance tools/sim_test_performance.cpp)
add_executable(kinect_sim_terminal_demo tools/sim_terminal_demo.cpp)
target_link_libraries (kinect_sim_viewer ${PROJECT_NAME})
target_link_libraries (kinect_sim_test_simple ${PROJECT_NAME})
target_link_libraries (kinect_sim_test_performance ${PROJECT_NAME})
target_link_libraries (kinect_sim_terminal_demo ${PROJECT_NAME})
