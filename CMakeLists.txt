cmake_minimum_required(VERSION 3.0)
project(diplom)

find_package(OpenCV)
find_package(OpenCL)

aux_source_directory( src SOURCES)
include_directories(${OpenCV_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIR})
add_executable(${PROJECT_NAME} ${SOURCES} )
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${OpenCL_LIBRARY})
