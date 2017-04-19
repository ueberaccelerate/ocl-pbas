cmake_minimum_required(VERSION 3.0)
project(diplom)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-std=c++11)

find_package(OpenCL)
find_package(OpenCV)
file ( GLOB_RECURSE CL_SOURCES src/*.cl)
file ( GLOB_RECURSE HPP_SOURCES src/*.hpp)
aux_source_directory( src SOURCES)

include_directories(${OpenCV_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIR})

add_executable(${PROJECT_NAME} ${SOURCES} ${CL_SOURCES} )
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${OpenCL_LIBRARY})
