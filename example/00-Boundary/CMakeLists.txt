add_executable(00-boundary main.cu ../../cuda/core/Scene.cu)
target_link_libraries(00-boundary ${PROJECT_NAME})
set_target_properties(00-boundary PROPERTIES CUDA_ARCHITECTURES "50")