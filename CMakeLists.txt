cmake_minimum_required(VERSION 3.2)

project(pose-estimation)

# Compiler setting
# Todo...
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-std=c++11 -march=native -O0 -g")

# set cmake module path
# list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# set output path
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# Eigen
include_directories("/usr/include/eigen3")

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# Include directories
include_directories(SYSTEM ${CMAKE_BINARY_DIR}) # Add current build directory as include directory as it contains generated files
include_directories(SYSTEM /usr/include)
include_directories(SYSTEM /usr/local/include)
include_directories(${CMAKE_SOURCE_DIR}/include)

# Generate static libs
add_library(${PROJECT_NAME}-core STATIC
    ${CMAKE_SOURCE_DIR}/src/config.cpp ...)

# Link libs
set(LIBRARIES ${LIBRARIES} ${Eigen3_LIBS} ${Sophus_LIBS} ${OpenCV_LIBS} ${CERES_LIBRARIES})

# Create executable
add_executable(${PROJECT_NAME} ${CMAKE_SOURCE_DIR}/src/${PROJECT_NAME}.cpp)
target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}-core ${LIBRARIES})

# Enable unit testing
enable_testing()
add_executable(${PROJECT_NAME}-test ${CMAKE_SOURCE_DIR}/test/vo.cpp)
target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME}-core ${LIBRARIES})
add_test(NAME ${PROJECT_NAME}-test COMMAND ${PROJECT_NAME}-test)

# Install executable
# install(TARGETS ${PROJECT_NAME} DESTINATION bin COMPONENT ${PROJECT_NAME})