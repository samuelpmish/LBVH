include(FetchContent)
FetchContent_Declare(
  wll_interface
  GIT_REPOSITORY https://github.com/samuelpmish/wll_interface.git
  GIT_TAG main
)
FetchContent_MakeAvailable(wll_interface)

add_library(bindings SHARED bindings.cpp)
target_link_libraries(bindings PUBLIC wll_interface)
