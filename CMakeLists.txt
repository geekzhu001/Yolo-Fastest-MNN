cmake_minimum_required(VERSION 3.10)
project(yolo)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)
include_directories(
        include/MNN
        include
)

link_directories(lib/)

add_executable(yolo src/main.cpp)
target_link_libraries(yolo MNN ${OpenCV_LIBS})
