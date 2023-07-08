cmake_minimum_required(VERSION 3.16)

project(lbvh_proj LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

file(GLOB lbvh_headers ${PROJECT_SOURCE_DIR}/cpp/*.hpp)
file(GLOB lbvh_sources ${PROJECT_SOURCE_DIR}/cpp/*.cpp)
add_library(libLBVH ${lbvh_sources} ${lbvh_headers})
target_include_directories(libLBVH PUBLIC ${PROJECT_SOURCE_DIR}/cpp)
set_target_properties(libLBVH PROPERTIES PREFIX "")

if(UNIX AND NOT APPLE)
  set_property(TARGET libLBVH PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

if (LBVH_ENABLE_PYTHON_BINDINGS)
  add_subdirectory(python)
endif()

if (LBVH_ENABLE_MATHEMATICA_BINDINGS)
  add_subdirectory(mathematica)
endif()

if (LBVH_ENABLE_TESTING)
  add_subdirectory(tests)
endif()
