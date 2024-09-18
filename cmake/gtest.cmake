if (NOT TARGET gtest_main)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.tar.gz
    GIT_SHALLOW  TRUE
    GIT_PROGRESS TRUE
  )

  message("resolving dependencies: gtest")
  FetchContent_MakeAvailable(googletest)
endif()
