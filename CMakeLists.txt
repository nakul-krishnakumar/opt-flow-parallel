cmake_minimum_required(VERSION 3.10)
project(opt-flow-parallel)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} include)

# Source files
file(GLOB SOURCES src/*.cpp)

add_executable(opt-flow-parallel main.cpp ${SOURCES})
target_link_libraries(opt-flow-parallel ${OpenCV_LIBS})
