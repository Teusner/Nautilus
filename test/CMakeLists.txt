## Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(event_test ../src/Event.cpp TestEvent.cu)
target_link_libraries(event_test gtest_main)
add_test(EventTest event_test)

add_executable(field_test ../src/Field.cu TestField.cu)
target_link_libraries(field_test gtest_main)
add_test(FieldTest field_test)

add_executable(emitter_test ../src/Module.cu TestEmitter.cu)
target_link_libraries(emitter_test gtest_main)
add_test(EmitterTest emitter_test)

add_executable(reciever_test ../src/Module.cu TestReciever.cu)
target_link_libraries(reciever_test gtest_main)
add_test(RecieverTest reciever_test)