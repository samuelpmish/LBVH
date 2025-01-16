if (NOT TARGET fm)
  FetchContent_Declare(
    TPL_FM
    GIT_REPOSITORY https://github.com/samuelpmish/fm.git
    GIT_TAG main
    GIT_SHALLOW  TRUE
    GIT_PROGRESS TRUE
  )

  message("resolving dependencies: fm")
  FetchContent_MakeAvailable(TPL_FM)
endif()
