enable_testing()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.tar.gz
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

#FetchContent_Declare(
#  dep_benchmark
#  URL https://github.com/google/benchmark/archive/refs/tags/v1.8.1.tar.gz
#)
## For Windows: Prevent overriding the parent project's compiler/linker settings
##set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
#set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
#FetchContent_MakeAvailable(dep_benchmark)

file(GLOB cpp_tests ${PROJECT_SOURCE_DIR}/tests/*.cpp)

foreach(filename ${cpp_tests})		

  get_filename_component(testname ${filename} NAME_WE)

  add_executable(${testname} ${filename})
  target_link_libraries(${testname} PUBLIC LBVH GTest::gtest_main)

  add_test(${testname} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${testname})		

endforeach(filename ${cpp_tests})