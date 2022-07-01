cmake_minimum_required(VERSION 3.16)
project(Nautilus LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

message("CXX compiler stuff:")
message(STATUS "CMAKE_CXX_COMPILER:         ${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_CXX_COMPILER_ID:      ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}")
message("CUDA compiler stuff:")
message(STATUS "CMAKE_CUDA_COMPILER:         ${CMAKE_CUDA_COMPILER}")
message(STATUS "CMAKE_CUDA_COMPILER_ID:      ${CMAKE_CUDA_COMPILER_ID}")
message(STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")

## Required software
find_package(CUDA REQUIRED)

## Sources and headers
include_directories(${CUDA_INCLUDE_DIRS} include cuda)

## C++ compiler options
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

## NVCC (CUDA compiler) options
set(CUDA_NVCC_FLAGS -gencode=arch=compute_60,code=compute_60 CACHE STRING "NVidia CUDA compiler flags" FORCE)
set(CUDA_NVCC_FLAGS_DEBUG -g -G -O0)
set(CUDA_NVCC_FLAGS_RELEASE -O3 -lineinfo)

## Thrust configuration
set(THRUST_HOST_SYSTEM CPP)
set(THRUST_DEVICE_SYSTEM CUDA)
set(THRUST_CPP_DIALECT 14)

## Source file containing the "main" function
set(MAIN_SOURCES src/main.cpp)
## Specify a name for the generated executable file
set(MAIN_EXE_NAME Nautilus)

################################################################################
# Looking for CXXOPTS
################################################################################
add_subdirectory(extern/cxxopts)

################################################################################
# Looking for xtensor
################################################################################
find_package(xtensor REQUIRED)

## 1. Compile...
add_library(${PROJECT_NAME} INTERFACE)
target_link_libraries(${PROJECT_NAME} INTERFACE xtensor)
target_include_directories(${PROJECT_NAME} INTERFACE ${xtensor_INCLUDE_DIRS} cuda/core)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "60")

## Tests
enable_testing()
add_subdirectory(test)

## Cuda
add_subdirectory(cuda)

## Example
add_subdirectory(example)

## 2. Link...
# target_link_libraries(${MAIN_EXE_NAME} ${NVTX_LIBRARY} ${CUDA_CUDART_LIBRARY})

## 3. Install...
# install(TARGETS ${MAIN_EXE_NAME} RUNTIME DESTINATION bin)
