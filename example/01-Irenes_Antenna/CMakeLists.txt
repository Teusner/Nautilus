add_executable(00-Irene main.cu ../../cuda/core/Scene.cu)
target_link_libraries(00-Irene ${PROJECT_NAME})
set_target_properties(00-Irene PROPERTIES CUDA_ARCHITECTURES "50")