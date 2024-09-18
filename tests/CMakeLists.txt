enable_testing()

include(FetchContent)


file(GLOB cpp_tests ${PROJECT_SOURCE_DIR}/tests/*.cpp)

foreach(filename ${cpp_tests})		

  get_filename_component(testname ${filename} NAME_WE)

  add_executable(${testname} ${filename})
  target_link_libraries(${testname} PUBLIC libLBVH GTest::gtest_main)

  add_test(${testname} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${testname})		

endforeach(filename ${cpp_tests})