cmake_minimum_required(VERSION 3.18)
project(yolov11_ocsort LANGUAGES CXX CUDA)

# ------------------ C++ & CUDA 设置 ------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# RTX 2080 Ti架构
set(CMAKE_CUDA_ARCHITECTURES 75)
message(STATUS "Compiling for CUDA architecture: ${CMAKE_CUDA_ARCHITECTURES}")

# CUDA编译设置
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# ------------------ 查找依赖 ------------------
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV found: ${OpenCV_VERSION}")

find_package(CUDA REQUIRED)
message(STATUS "CUDA found: ${CUDA_VERSION}")

# 查找Eigen3
find_path(EIGEN3_INCLUDE_DIR NAMES Eigen/Core
    PATHS
    /usr/include/eigen3
    /usr/local/include/eigen3
    /usr/include
)

if(EIGEN3_INCLUDE_DIR)
    message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "Eigen3 not found")
endif()

# ------------------ TensorRT 设置 ------------------
# AutoDL环境中的TensorRT路径
set(TENSORRT_ROOT /root/CUDA/TensorRT-8.6.1.6)

if(EXISTS ${TENSORRT_ROOT})
    message(STATUS "TensorRT found at: ${TENSORRT_ROOT}")
    
    # 添加TensorRT头文件路径
    include_directories(
        ${TENSORRT_ROOT}/include
        ${TENSORRT_ROOT}/include/NvInfer
        ${TENSORRT_ROOT}/include/NvInferPlugin
    )
    
    # 添加TensorRT库路径
    link_directories(${TENSORRT_ROOT}/targets/x86_64-linux-gnu/lib)
    
    # 设置TensorRT库文件
    set(TENSORRT_LIBRARIES 
        nvinfer
        nvinfer_plugin
    )
    
    message(STATUS "TensorRT libraries: ${TENSORRT_LIBRARIES}")
else()
    message(FATAL_ERROR "TensorRT not found at ${TENSORRT_ROOT}")
endif()

# ------------------ Include 目录 ------------------
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/detector/include
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

# ------------------ OCSort 跟踪器库 ------------------
add_library(ocsort_tracker STATIC
    src/Association.cpp
    src/KalmanBoxTracker.cpp
    src/KalmanFilter.cpp
    src/lapjv.cpp
    src/OCSort.cpp
    src/Utilities.cpp
)

target_include_directories(ocsort_tracker PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ocsort_tracker PUBLIC
    ${OpenCV_LIBS}
)

# ------------------ YOLOv11 检测器库 ------------------
add_library(yolov11_detector STATIC
    detector/src/yolov11.cpp
    detector/src/preprocess.cu
    detector/src/postprocess.cu
)

set_target_properties(yolov11_detector PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(yolov11_detector PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/detector/include
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(yolov11_detector PUBLIC
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    cudart
    ${TENSORRT_LIBRARIES}
)

# ------------------ 主可执行文件 ------------------
add_executable(${PROJECT_NAME}
    main.cpp
)

# ------------------ 链接所有库 ------------------
target_link_libraries(${PROJECT_NAME} PRIVATE
    yolov11_detector
    ocsort_tracker
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBRARIES}
    cudart
    pthread
)

# ------------------ 编译选项 ------------------
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
)

target_compile_options(yolov11_detector PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)