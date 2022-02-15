add_executable(example main.cu ../cuda/core/Scene.cu ../cuda/utils/constant_memory.cu)
target_link_libraries(example export)

add_subdirectory(00-Boundary)