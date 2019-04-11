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

project(preintegrate)

################################################################################
# This project requires C++14 or newer.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Build a static binary.
set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")
set(CMAKE_CXX_FLAGS "-O0 -g")
# # Add further warning levels.
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

################################################################################
# Include directories
include_directories(SYSTEM /usr/include)
include_directories(SYSTEM /usr/local/include)
include_directories(${CMAKE_SOURCE_DIR}/include)
# Add current build directory as include directory as it contains generated files.
include_directories(SYSTEM ${CMAKE_BINARY_DIR})

################################################################################
# Gather all object code first to avoid double compilation.
# Eigen
include_directories("/usr/local/include/eigen3")
set(LIBRARIES ${LIBRARIES} ${Eigen3_LIBS})

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})
set(LIBRARIES ${LIBRARIES} ${Sophus_LIBS})

# OpenCV
# find_package(OpenCV REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS})
# set(LIBRARIES ${LIBRARIES} ${OpenCV_LIBS})

# Ceres
# find_package(Ceres REQUIRED)
# include_directories(${CERES_INCLUDE_DIRS})
# set(LIBRARIES ${LIBRARIES} ${CERES_LIBRARIES})

################################################################################
# Create executable
add_executable(${PROJECT_NAME} ${CMAKE_SOURCE_DIR}/${PROJECT_NAME}.cpp)
target_link_libraries(${PROJECT_NAME} ${LIBRARIES})

################################################################################
# Enable unit testing
# enable_testing()
# add_executable(${PROJECT_NAME}-test ${CMAKE_SOURCE_DIR}/test/vo.cpp)
# target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME}-core ${LIBRARIES})
# add_test(NAME ${PROJECT_NAME}-test COMMAND ${PROJECT_NAME}-test)

################################################################################
# Install executable
# install(TARGETS ${PROJECT_NAME} DESTINATION bin COMPONENT ${PROJECT_NAME})