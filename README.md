# LBVH

An implementation of the LBVH algorithm for performing fast bounding-box intersections.
NVIDIA has a great series of blog posts that describes, in detail, the algorithm for LBVH construction/traversal:

1. https://developer.nvidia.com/blog/thinking-parallel-part-i-collision-detection-gpu/
2. https://developer.nvidia.com/blog/thinking-parallel-part-ii-tree-traversal-gpu/
3. https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/

This repo's implementation is CPU-only for now, but it does include python and Mathematica bindings.

## Python

Precompiled binaries are available for many versions of python on mac, linux and windows. 
To install python bindings for use with `jax` or `numpy`, do

```sh
pip3 install lbvh
```

The python bindings expect arrays of bounding boxes to be specified in one of the following formats:

```py
# list of 2D bounding boxes, as arrays with shape (*, 4)
[[min_x, min_y, max_x, max_y], ... ]

# list of 2D bounding boxes, as arrays with shape (*, 2, 2)
[[[min_x, min_y], [max_x, max_y]], ... ]

# list of 3D bounding boxes, as arrays with shape (*, 6)
[[min_x, min_y, min_z, max_x, max_y, max_z], ... ]

# list of 3D bounding boxes, as arrays with shape (*, 2, 3)
[[[min_x, min_y, min_z], [max_x, max_y, max_z]], ... ]
```

See https://github.com/samuelpmish/LBVH/tree/main/python for some basic examples of bounding box queries.


## C++

To build from source:

### 1. clone the repo 
```sh
git clone git@github.com:samuelpmish/LBVH.git
```

### 2. configure CMake 
```sh
cd LBVH
cmake . -Bbuild -DCMAKE_BUILD_TYPE=Release 
```
This project has a few (off by default) configure arguments that control which targets get built, including
- `LBVH_ENABLE_PYTHON_BINDINGS`
- `LBVH_ENABLE_MATHEMATICA_BINDINGS`
- `LBVH_ENABLE_TESTS`

### 3. build
```sh
cmake --build build
```

----

To use LBVH in another CMake project, you can use `FetchContent` to automatically build it along with the parent CMake project

```cmake

...

# fetch and build LBVH
include(FetchContent)
FetchContent_Declare(
  LBVH
  GIT_REPOSITORY https://github.com/samuelpmish/LBVH.git
  GIT_TAG main
)
set(LBVH_ENABLE_PYTHON_BINDINGS OFF CACHE INTERNAL "")
set(LBVH_ENABLE_MATHEMATICA_BINDINGS OFF CACHE INTERNAL "")
set(LBVH_ENABLE_TESTING OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(LBVH)

...

# link it against any targets in the parent project
target_link_libraries(my_project PUBLIC LBVH)
```
