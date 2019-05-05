# Copyright (C) 2018  Christian Berger
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

cmake_minimum_required(VERSION 3.2)

option(FOR_CFSD "cfsd state estimation" OFF)
if(FOR_CFSD)
    add_definitions(-DCFSD)
    project(cfsd-state-estimation)
endif()

option(FOR_EUROC "euroc state estimation" ON)
if(FOR_EUROC)
    add_definitions(-DEUROC)
    project(euroc-state-estimation)
endif()

option(FOR_KITTI "kitti state estimation" OFF)
if(FOR_KITTI)
    add_definitions(-DKITTI)
    project(kitti-state-estimation)
endif()

################################################################################
# # Defining the relevant versions of OpenDLV Standard Message Set and libcluon.
set(OPENDLV_STANDARD_MESSAGE_SET opendlv-standard-message-set-v0.9.7.odvd)
set(CLUON_COMPLETE cluon-complete-v0.0.121.hpp)

################################################################################
# Set the search path for .cmake files.
# set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}" ${CMAKE_MODULE_PATH})
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

################################################################################
# This project requires C++14 or newer.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Build a static binary.
set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")
set(CMAKE_CXX_FLAGS "-O0 -g")
# Add further warning levels.
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
#     -D_XOPEN_SOURCE=700 \
#     -D_FORTIFY_SOURCE=2 \
#     -O2 \
#     -fstack-protector \
#     -fomit-frame-pointer \
#     -pipe \
#     -Weffc++ \
#     -Wall -Wextra -Wshadow -Wdeprecated \
#     -Wdiv-by-zero -Wfloat-equal -Wfloat-conversion -Wsign-compare -Wpointer-arith \
#     -Wuninitialized -Wunreachable-code \
#     -Wunused -Wunused-function -Wunused-label -Wunused-parameter -Wunused-but-set-parameter -Wunused-but-set-variable \
#     -Wunused-value -Wunused-variable -Wunused-result \
#     -Wmissing-field-initializers -Wmissing-format-attribute -Wmissing-include-dirs -Wmissing-noreturn")
# Threads are necessary for linking the resulting binaries as UDPReceiver is running in parallel.
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

################################################################################
# Set debug flag
option(WITH_DEBUG_IMG "Debug mode for feature-tracker" OFF)
if(WITH_DEBUG_IMG)
    add_definitions(-DDEBUG_IMG)
endif()

option(WITH_DEBUG_IMU "Debug mode for imu-preintegrator" OFF)
if(WITH_DEBUG_IMU)
    add_definitions(-DDEBUG_IMU)
endif()

################################################################################
# Set output path
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

################################################################################
# Extract cluon-msc from cluon-complete.hpp.
# ${CMAKE_BINARY_DIR} is the state-estimation/build/ directory
add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/cluon-msc
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_SOURCE_DIR}/include/${CLUON_COMPLETE} ${CMAKE_BINARY_DIR}/cluon-complete.hpp
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_BINARY_DIR}/cluon-complete.hpp ${CMAKE_BINARY_DIR}/cluon-complete.cpp
    COMMAND ${CMAKE_CXX_COMPILER} -o ${CMAKE_BINARY_DIR}/cluon-msc ${CMAKE_BINARY_DIR}/cluon-complete.cpp -std=c++14 -pthread -D HAVE_CLUON_MSC
    DEPENDS ${CMAKE_SOURCE_DIR}/include/${CLUON_COMPLETE})
################################################################################
# Generate opendlv-standard-message-set.hpp from ${OPENDLV_STANDARD_MESSAGE_SET} file.
add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/opendlv-standard-message-set.hpp
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMAND ${CMAKE_BINARY_DIR}/cluon-msc --cpp --out=${CMAKE_BINARY_DIR}/opendlv-standard-message-set.hpp ${CMAKE_SOURCE_DIR}/include/${OPENDLV_STANDARD_MESSAGE_SET}
    DEPENDS ${CMAKE_SOURCE_DIR}/include/${OPENDLV_STANDARD_MESSAGE_SET} ${CMAKE_BINARY_DIR}/cluon-msc)

################################################################################
# Include directories
include_directories(SYSTEM /usr/include)
include_directories(SYSTEM /usr/local/include)
include_directories(${CMAKE_SOURCE_DIR}/include)
# Add current build directory as include directory as it contains generated files.
include_directories(SYSTEM ${CMAKE_BINARY_DIR})

################################################################################
# Gather all object code first to avoid double compilation.
set(LIBRARIES Threads::Threads)

if(UNIX)
    if(NOT "${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
        find_package(LibRT REQUIRED)
        set(LIBRARIES ${LIBRARIES} ${LIBRT_LIBRARIES})
        include_directories(SYSTEM ${LIBRT_INCLUDE_DIR})
    endif()
endif()

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

# Pangolin
option(WITH_VIEWER "Enable visulization" ON)
if (WITH_VIEWER)
    add_definitions(-DUSE_VIEWER)
    find_package(Pangolin 0.4 REQUIRED)
    include_directories(${Pangolin_INCLUDE_DIRS})
    set(LIBRARIES ${LIBRARIES} ${Pangolin_LIBRARIES})
#else()
endif()

################################################################################
# Generate static libs
add_library(state-estimation-core STATIC
    ${CMAKE_SOURCE_DIR}/src/config.cpp
    ${CMAKE_SOURCE_DIR}/src/optimizer.cpp
    ${CMAKE_SOURCE_DIR}/src/imu-preintegrator.cpp
    ${CMAKE_SOURCE_DIR}/src/feature-tracker.cpp
    ${CMAKE_SOURCE_DIR}/src/map.cpp
    ${CMAKE_SOURCE_DIR}/src/visual-inertial-slam.cpp
    ${CMAKE_SOURCE_DIR}/src/viewer.cpp
    ${CMAKE_SOURCE_DIR}/src/ORBextractor.cc)

################################################################################
# Create executable
add_executable(${PROJECT_NAME} ${CMAKE_SOURCE_DIR}/src/${PROJECT_NAME}.cpp ${CMAKE_BINARY_DIR}/opendlv-standard-message-set.hpp)
target_link_libraries(${PROJECT_NAME} state-estimation-core ${LIBRARIES})

################################################################################
# Enable unit testing
# enable_testing()
# add_executable(${PROJECT_NAME}-test ${CMAKE_SOURCE_DIR}/test/vo.cpp)
# target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME}-core ${LIBRARIES})
# add_test(NAME ${PROJECT_NAME}-test COMMAND ${PROJECT_NAME}-test)

################################################################################
# Install executable
# install(TARGETS ${PROJECT_NAME} DESTINATION bin COMPONENT ${PROJECT_NAME})