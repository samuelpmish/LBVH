if (NOT TARGET nanobind) 
  FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_SHALLOW  TRUE
    GIT_PROGRESS TRUE
  )

  message("resolving dependency: nanobind")
  FetchContent_MakeAvailable(nanobind)
endif()