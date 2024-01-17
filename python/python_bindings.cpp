#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <array>
#include <vector>
#include <iostream>

#include "BVH.hpp"

namespace nb = nanobind;

using namespace nb::literals;

template < int dim >
std::vector< AABB<dim> > convert_matrices_to_AABBs(nb::ndarray< nb::numpy, const float, nb::shape<nb::any, 2, dim> > arr) {
  size_t n = arr.shape(0);
  std::vector< AABB<dim> > output(n);
  for (size_t i = 0; i < n; i++) {
    AABB<dim> box;

    if constexpr (dim == 2) {
      box.min[0] = arr(i, 0, 0); box.min[1] = arr(i, 0, 1);
      box.max[0] = arr(i, 1, 0); box.max[1] = arr(i, 1, 1);
    }

    if constexpr (dim == 3) {
      box.min[0] = arr(i, 0, 0); box.min[1] = arr(i, 0, 1); box.min[2] = arr(i, 0, 2);
      box.max[0] = arr(i, 1, 0); box.max[1] = arr(i, 1, 1); box.max[2] = arr(i, 1, 2);
    }

    output[i] = box;
  }
  return output;
}

template < int dim >
std::vector< AABB<dim> > convert_vectors_to_AABBs(nb::ndarray< nb::numpy, const float, nb::shape<nb::any, 2 * dim> > arr) {
  size_t n = arr.shape(0);
  std::vector< AABB<dim> > output(n);
  for (size_t i = 0; i < n; i++) {
    AABB<dim> box;

    if constexpr (dim == 2) {
      box.min[0] = arr(i, 0); box.min[1] = arr(i, 1);
      box.max[0] = arr(i, 2); box.max[1] = arr(i, 3);
    }

    if constexpr (dim == 3) {
      box.min[0] = arr(i, 0); box.min[1] = arr(i, 1); box.min[2] = arr(i, 2);
      box.max[0] = arr(i, 3); box.max[1] = arr(i, 4); box.max[2] = arr(i, 5);
    }

    output[i] = box;
  }
  return output;
}

template < int dim >
nb::ndarray< nb::numpy, int, nb::shape<nb::any, 2> > find_pairs(const std::vector< AABB<dim> > & boxes) {
  BVH<dim> bvh(boxes);

  std::vector< std::array< int, 2 > > pairs;
  for (int i = 0; i < boxes.size(); i++) {
    bvh.query(boxes[i], [&](int j){ if (i < j) pairs.push_back({i, j}); });
  }

  // Delete 'data' when the 'owner' capsule expires
  int * data = new int[pairs.size() * 2];
  nb::capsule owner(data, [](void *p) noexcept {
     delete[] (float *) p;
  });
  
  nb::ndarray<nb::numpy, int, nb::shape<nb::any, 2> > output(data, {pairs.size(), 2}, owner);
  for (int i = 0; i < pairs.size(); i++) {
    output(i, 0) = pairs[i][0];
    output(i, 1) = pairs[i][1];
  }

  return output;
}

NB_MODULE(LBVH, m) {
  m.def("find_intersections", [](nb::ndarray< nb::numpy, const float, nb::shape<nb::any, 4> > boxes) {
    return find_pairs(convert_vectors_to_AABBs<2>(boxes));
  });

  m.def("find_intersections", [](nb::ndarray< nb::numpy, const float, nb::shape<nb::any, 6> > boxes) {
    return find_pairs(convert_vectors_to_AABBs<3>(boxes));
  });

  m.def("find_intersections", [](nb::ndarray< nb::numpy, const float, nb::shape<nb::any, 2, 2> > boxes) {
    return find_pairs(convert_matrices_to_AABBs<2>(boxes));
  });

  m.def("find_intersections", [](nb::ndarray< nb::numpy, const float, nb::shape<nb::any, 2, 3> > boxes) {
    return find_pairs(convert_matrices_to_AABBs<3>(boxes));
  });
}
