add_executable(00-boundary main.cu ../../cuda/core/Scene.cu ../../cuda/utils/constant_memory.cu)
target_link_libraries(00-boundary export)