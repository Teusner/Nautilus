add_executable(example main.cu)
target_link_libraries(example ${PROJECT_NAME})
set_target_properties(example PROPERTIES CUDA_ARCHITECTURES "50")

add_subdirectory(00-Boundary)