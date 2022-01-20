cmake_minimum_required(VERSION 3.16)
project(Nautilus LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 11)

## Required software
find_package(CUDA REQUIRED)

## Cuda require gcc-8
# set(CMAKE_CXX_COMPILER /usr/bin/g++-7)

## Sources and headers
include_directories(${CUDA_INCLUDE_DIRS} include cuda)

## Split the source into a GPU part (compiled by nvcc) and a
## CPU part (compiled by e.g. gcc).
## (This may not always be necessary.)
file(GLOB CPU_SOURCES src/*.cpp)
file(GLOB GPU_SOURCES src/*.cu)

## C++ compiler options
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

## NVCC (CUDA compiler) options
# set(CUDA_NVCC_FLAGS -gencode=arch=compute_30,code=sm_30 CACHE STRING "NVidia CUDA compiler flags" FORCE)
set(CUDA_NVCC_FLAGS_DEBUG -g -G -O0)
set(CUDA_NVCC_FLAGS_RELEASE -O3 -lineinfo)

## Source file containing the "main" function
set(MAIN_SOURCES src/main.cpp)
## Specify a name for the generated executable file
set(MAIN_EXE_NAME Nautilus)

## 1. Compile...
add_executable(nautilus src/main.cu src/Module.cu src/Scene.cpp cuda/core/Material.cu src/Event.cpp cuda/utils/constant_memory.cu)

## Tests
enable_testing()
add_subdirectory(test)

## Cuda
add_subdirectory(cuda)

## 2. Link...
# target_link_libraries(${MAIN_EXE_NAME} ${NVTX_LIBRARY} ${CUDA_CUDART_LIBRARY})

## 3. Install...
# install(TARGETS ${MAIN_EXE_NAME} RUNTIME DESTINATION bin)
