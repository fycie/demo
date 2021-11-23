cmake_minimum_required(VERSION 2.8)
SET(CMAKE_C_COMPILER g++)
add_compile_options(-std=c++11)
project( gms_match_demo)

set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# OpenCV 
find_package( OpenCV REQUIRED )

# 
include_directories( 
 include
${OpenCV_INCLUDE_DIRS} 
)


add_executable( gms_match_demo ./src/gms+ransac.cpp )
target_link_libraries( gms_match_demo ${OpenCV_LIBS} )
