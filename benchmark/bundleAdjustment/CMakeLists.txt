cmake_minimum_required( VERSION 2.8 )
project( bundleAdjustment )

set( CMAKE_CXX_FLAGS "-std=c++11" )

# Eigen
include_directories("/usr/include/eigen3")
set(LIBRARIES ${LIBRARIES} ${Eigen3_LIBS})

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})
set(LIBRARIES ${LIBRARIES} ${Sophus_LIBS})

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(LIBRARIES ${LIBRARIES} ${OpenCV_LIBS})

# Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})
set(LIBRARIES ${LIBRARIES} ${CERES_LIBRARIES})

add_executable( ${PROJECT_NAME} ${PROJECT_NAME}.cpp )
target_link_libraries( ${PROJECT_NAME} ${LIBRARIES} )