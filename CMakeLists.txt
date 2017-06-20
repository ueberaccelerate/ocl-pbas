cmake_minimum_required(VERSION 3.0)
project(diplom)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-std=c++11 -O3)

set(argagg_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/argagg/include)

find_package(OpenCL)
find_package(OpenCV)

file(COPY ${CMAKE_SOURCE_DIR}/.clang-format DESTINATION ${CMAKE_BINARY_DIR})

file ( GLOB_RECURSE CL_SOURCES src/*.cl)
file ( GLOB_RECURSE HPP_SOURCES src/*.hpp)
file ( GLOB_RECURSE H_SOURCES src/*.h)
file ( GLOB_RECURSE CPP_SOURCES src/*.cpp)

aux_source_directory( src SOURCES)

include_directories(${OpenCV_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIRS} ${argagg_INCLUDE_DIR})

add_executable(detector ${SOURCES} ${CL_SOURCES} ${HPP_SOURCES} ${H_SOURCES} ${CPP_SOURCES})
target_link_libraries(detector ${OpenCV_LIBS} ${OpenCL_LIBRARY})
