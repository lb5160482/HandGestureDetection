cmake_minimum_required(VERSION 2.8.12)

set(PROJECT_NAME HandGesture)
project(${PROJECT_NAME})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wall -std=c++11 -fPIC")
find_package(OpenCV REQUIRED)

include_directories(
	${OpenCV_INCLUDE_DIRS}
	${HandGesture_SOURCE_DIR}/include
)
add_library(SkinDetector
	${HandGesture_SOURCE_DIR}/include/SkinDetector.h
	${HandGesture_SOURCE_DIR}/src/SkinDetector.cpp
	)

add_executable(main src/main.cpp)

target_link_libraries(main
    ${OpenCV_LIBS}
    SkinDetector
)
