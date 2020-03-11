cmake_minimum_required(VERSION 2.8)
project(object_recognition_node)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-std=c++11 -pthread")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(SPECIAL_OS_LIBS "pthread" "X11" "z")

#include(CMakeForceCompiler)
#CMAKE_FORCE_CXX_COMPILER(mpicxx ${CMAKE_CXX_COMPILER_ID})

set (MPI_ROOT /usr/include/openmpi/)
include_directories(${MPI_ROOT})

#set (MPI_ROOT /media/aditya/A69AFABA9AFA85D9/Cruzr/code/openmpi-4.0.0/install)
include(CMakeForceCompiler)
#CMAKE_FORCE_CXX_COMPILER(${MPI_ROOT}/bin/mpicxx ${CMAKE_CXX_COMPILER_ID})
CMAKE_FORCE_CXX_COMPILER(/usr/bin/mpicxx ${CMAKE_CXX_COMPILER_ID})
#include_directories(${MPI_ROOT}/include)

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

include_directories(${CUDA_INCLUDE_DIRS})

find_package(catkin
  REQUIRED
  genmsg actionlib_msgs actionlib
  COMPONENTS
  image_transport
  roscpp sensor_msgs perception_utils
  tf tf_conversions eigen_conversions sbpl_perception
)
find_package(Threads REQUIRED)
set(THREADS_PREFER_PTHREAD_FLAG ON)
if(THREADS_HAVE_PTHREAD_ARG)
  target_compile_options(${PROJECT_NAME} PUBLIC "-pthread")
endif()

#find_package(Boost COMPONENTS mpi REQUIRED)
#find_package(MPI REQUIRED)

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

find_package(Boost 1.58.0 REQUIRED thread mpi serialization system)
find_package(ZLIB)

find_package(Boost 1.58.0 REQUIRED mpi serialization system)

# find_package(catkin REQUIRED genmsg actionlib_msgs actionlib)
add_action_files(DIRECTORY action FILES DoPerch.action)
add_service_files(FILES LocalizeObjects.srv)

include_directories(${PROJECT_SOURCE_DIR}/include ${catkin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
#include_directories(MPI_INCLUDE_PATH)

generate_messages(
  DEPENDENCIES
  actionlib_msgs
  std_msgs
  geometry_msgs
  sensor_msgs
  tf
  )

catkin_package(
  CATKIN_DEPENDS image_transport roscpp sensor_msgs perception_utils tf tf_conversions
  sbpl_perception
  DEPENDS
  #MPI Boost
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  )

add_library(${PROJECT_NAME}
  src/perception_interface.cpp)
target_link_libraries(${PROJECT_NAME} ${MPI_LIBRARIES} ${catkin_LIBRARIES} ${PCL_LIBRARIES}
  ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${ZLIB_LIBRARIES} ${CUDA_LIBRARIES})


add_executable(obj_rec_node src/object_recognition_node.cpp)
target_link_libraries(obj_rec_node ${PROJECT_NAME}
  #libvtkCommon.so libvtkFiltering.so libvtkRendering.so libvtkIO.so
  #libvtkGraphics.so
  )

add_executable(object_localizer src/object_localizer_service.cpp)
target_link_libraries(object_localizer ${catkin_LIBRARIES} ${PROJECT_NAME}
  #libvtkCommon.so libvtkFiltering.so libvtkRendering.so libvtkIO.so
  #libvtkGraphics.so
  )

add_executable(object_localizer_client_example
  src/test/object_localizer_client_example.cpp)
target_link_libraries(object_localizer_client_example ${catkin_LIBRARIES} ${PROJECT_NAME}
  #libvtkCommon.so libvtkFiltering.so libvtkRendering.so libvtkIO.so
  #libvtkGraphics.so
  )

#install(PROGRAMS
#  scripts/pr2_perch_client.py
#  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  PATTERN ".svn" EXCLUDE)

install(DIRECTORY rviz/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  PATTERN ".svn" EXCLUDE)

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS object_localizer
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS obj_rec_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
