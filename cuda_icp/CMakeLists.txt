cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 14)
project(cuda_icp)

option(USE_CUDA "use cuda or not" ON)

SET(CMAKE_BUILD_TYPE "Release")
#SET(CMAKE_BUILD_TYPE "Debug")

SET(CMAKE_CXX_FLAGS "-fopenmp -Wall -Wno-sign-compare -fPIC $ENV{CXXFLAGS}")
SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3")

# opencv
find_package(OpenCV 3 REQUIRED)
list(APPEND icp_inc ${OpenCV_INCLUDE_DIRS})
list(APPEND icp_lib ${OpenCV_LIBS})

if(USE_CUDA)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-8.0)
add_definitions(-DCUDA_ON)
endif()

if(USE_CUDA)
# cuda
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O3 --default-stream per-thread -Xcompiler -fopenmp")
list(APPEND icp_inc ${CUDA_INCLUDE_DIRS})
list(APPEND icp_lib ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
endif()


# eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories("include")

# open3d
find_package(Open3D REQUIRED)
if (Open3D_FOUND)
    message(STATUS "Found Open3D ${Open3D_VERSION}")

    include_directories(${Open3D_INCLUDE_DIRS})
    link_directories(${Open3D_LIBRARY_DIRS})

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${Open3D_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Open3D_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${Open3D_EXE_LINKER_FLAGS}")
else ()
    message(SEND_ERROR "Open3D not found")
endif ()
list(APPEND icp_lib ${Open3D_LIBRARIES})

# src
SET(icp_cuda_srcs  icp.cu scene/common.cu scene/depth_scene/depth_scene.cu scene/pcd_scene/pcd_scene.cu)
SET(icp_srcs  icp.cpp scene/common.cpp scene/depth_scene/depth_scene.cpp scene/pcd_scene/pcd_scene.cpp)


if(USE_CUDA)
CUDA_COMPILE(icp_cuda_objs ${icp_cuda_srcs})
endif()

find_package(catkin REQUIRED COMPONENTS roscpp OpenCV)

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
)

# lib & test exe
add_library(cuda_icp
               ${icp_srcs}
               ${icp_cuda_srcs}
               ${icp_cuda_objs} 
)
target_include_directories(cuda_icp PUBLIC ${icp_inc})
target_link_libraries(cuda_icp PUBLIC ${icp_lib}  ${catkin_LIBRARIES})

add_executable(icp_test test.cpp)
target_link_libraries(icp_test cuda_icp)

install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})

