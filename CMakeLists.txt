cmake_minimum_required(VERSION 3.18)
project(tensorrt_cpp_api)

# Use ccache to speed up rebuilds
include(cmake/ccache.cmake)

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-deprecated-declarations")

# For finding FindTensorRT.cmake
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# TensorRT installation directory hint
set(TensorRT_DIR "" CACHE PATH "TensorRT install directory.")
if (NOT TensorRT_DIR)
    message(WARNING "Finding TensorRT could be difficult for non-standard location, please specify TensorRT_DIR.")
endif()

# We require CUDA, OpenCV, and TensorRT
find_package(TensorRT 8.6 REQUIRED)
find_package(CUDA 11.7 REQUIRED)
find_package(OpenCV 4.8 REQUIRED)

# argparse
include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)

add_library(tensorrt_cpp_api SHARED src/engine.cpp)
# use rpath instead or runpath, because tensorrt libs has no runpath and rpath is transitive
# but carefull with LD_LIBRARY_PATH
if (UNIX AND NOT APPLE)
    target_link_options(tensorrt_cpp_api PUBLIC "-Wl,--disable-new-dtags")
endif()
target_include_directories(tensorrt_cpp_api PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
target_link_libraries(tensorrt_cpp_api PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})

add_executable(run_inference_benchmark src/benchmark.cpp)
target_link_libraries(run_inference_benchmark tensorrt_cpp_api)

add_executable(run_prediction_image src/main.cpp src/transforms.cpp)
target_link_libraries(run_prediction_image tensorrt_cpp_api argparse)