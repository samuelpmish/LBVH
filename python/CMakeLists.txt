include(FetchContent)
FetchContent_Declare(
  nanobind
  GIT_REPOSITORY git@github.com:wjakob/nanobind.git
)
FetchContent_MakeAvailable(nanobind)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
nanobind_add_module(pyLBVH python_bindings.cpp)
target_link_libraries(pyLBVH PUBLIC LBVH)