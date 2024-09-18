cmake_minimum_required(VERSION 3.16)

project(lbvh_proj LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# disable if you want FetchContent to grab the most recent
# dependencies when configuring cmake
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

file(GLOB lbvh_headers ${PROJECT_SOURCE_DIR}/cpp/*.hpp)
file(GLOB lbvh_sources ${PROJECT_SOURCE_DIR}/cpp/*.cpp)
add_library(libLBVH ${lbvh_sources} ${lbvh_headers})
target_include_directories(libLBVH PUBLIC ${PROJECT_SOURCE_DIR}/cpp)
set_target_properties(libLBVH PROPERTIES PREFIX "")

if(UNIX AND NOT APPLE)
  set_property(TARGET libLBVH PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

################
# dependencies #
################

include(FetchContent)
include(cmake/fm.cmake)

if (LBVH_ENABLE_PYTHON_BINDINGS)
  include(cmake/nanobind.cmake)
  add_subdirectory(python)
endif()

if (LBVH_ENABLE_MATHEMATICA_BINDINGS)
  include(cmake/wll_interface.cmake)
  add_library(LBVH_wll SHARED mathematica/bindings.cpp)
  target_link_libraries(LBVH_wll PUBLIC wll_interface)
endif()

if (LBVH_ENABLE_TESTING)
  add_subdirectory(tests)
endif()
