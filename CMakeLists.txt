cmake_minimum_required(VERSION 3.22)

project(snowStorm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV
find_package(OpenCV 4.10 REQUIRED)

message(STATUS "Found OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include directories: ${OpenCV_INCLUDE_DIRS}")



# Copy YOLO model into build directory
configure_file(
    ${CMAKE_SOURCE_DIR}/yolo11n.onnx
    ${CMAKE_BINARY_DIR}/yolo11n.onnx
    COPYONLY
)

# Executable
add_executable(
    snowStorm
    main.cpp
    cat.cpp
)

set(ONNXRUNTIME_DIR
    "/lib/onnxruntime-linux-x64-gpu-1.20.1"
)

target_include_directories(
    snowStorm PRIVATE
    "${ONNXRUNTIME_DIR}/include"
)

target_link_libraries(
    snowStorm PRIVATE
    ${OpenCV_LIBS}
    "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so"
)

set_target_properties(
    snowStorm PROPERTIES
    BUILD_RPATH
    "${ONNXRUNTIME_DIR}/lib;/usr/local/cuda-12.4/lib64"
)