if (NOT TARGET nanobind) 
  FetchContent_Declare(
    wll_interface
    GIT_REPOSITORY https://github.com/samuelpmish/wll_interface.git
    GIT_TAG main
    GIT_SHALLOW  TRUE
    GIT_PROGRESS TRUE
  )
  message("resolving dependency: wll_interface")
  FetchContent_MakeAvailable(wll_interface)
endif()