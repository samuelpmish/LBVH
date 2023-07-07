#pragma once

#include <array>
#include <string>
#include <nanobind/tensor.h>

#include "containers/ndarray.hpp"

namespace nb = nanobind;

using jax_tensor_1 = nanobind::tensor<nb::jax, double, nb::shape<nb::any> >;
using jax_tensor_2 = nanobind::tensor<nb::jax, double, nb::shape<nb::any, nb::any> >;
using jax_tensor_3 = nanobind::tensor<nb::jax, double, nb::shape<nb::any, nb::any, nb::any> >;
using jax_tensor_4 = nanobind::tensor<nb::jax, double, nb::shape<nb::any, nb::any, nb::any, nb::any> >;

using jax_tensor_1u = nanobind::tensor<nb::jax, uint64_t, nb::shape<nb::any> >;
using jax_tensor_2u = nanobind::tensor<nb::jax, uint64_t, nb::shape<nb::any, nb::any> >;

inline jax_tensor_1u create_jax_tensor_u64(size_t shape) {
  return jax_tensor_1u(new uint64_t[shape], 1, &shape);
};

inline jax_tensor_2u create_jax_tensor_u64(std::array<size_t, 2> shape) {
  return jax_tensor_2u(new uint64_t[shape[0] * shape[1]], 2, &shape[0]);
};

inline jax_tensor_1 create_jax_tensor(size_t shape) {
  return jax_tensor_1(new double[shape], 1, &shape);
};

inline jax_tensor_2 create_jax_tensor(size_t shape0, size_t shape1) {
  size_t shape[2] = {shape0, shape1};
  return jax_tensor_2(new double[shape0 * shape1], 2, shape);
};

inline jax_tensor_3 create_jax_tensor(size_t shape0, size_t shape1, size_t shape2) {
  size_t shape[3] = {shape0, shape1, shape2};
  return jax_tensor_3(new double[shape0 * shape1 * shape2], 3, shape);
};

inline jax_tensor_4 create_jax_tensor(size_t shape0, size_t shape1, size_t shape2, size_t shape3) {
  size_t shape[4] = {shape0, shape1, shape2, shape3};
  return jax_tensor_4(new double[shape0 * shape1 * shape2 * shape3], 4, shape);
};

inline ndview<double> to_ndview(jax_tensor_1 j) {
  return ndview<double>((double*)j.data(), {j.shape(0)}); 
}

inline ndview<double,2> to_ndview(jax_tensor_2 j) {
  return ndview<double,2>((double*)j.data(), {j.shape(0), j.shape(1)}); 
}

inline ndview<double,3> to_ndview(jax_tensor_3 j) {
  return ndview<double,3>((double*)j.data(), {j.shape(0), j.shape(1), j.shape(2)}); 
}

inline ndview<double,4> to_ndview(jax_tensor_4 j) {
  return ndview<double,4>((double*)j.data(), {j.shape(0), j.shape(1), j.shape(2), j.shape(3)}); 
}

inline ndview<uint64_t> to_ndview(jax_tensor_1u j) {
  return ndview<uint64_t>((uint64_t*)j.data(), {j.shape(0)}); 
}

inline ndview<uint64_t,2> to_ndview(jax_tensor_2u j) {
  return ndview<uint64_t,2>((uint64_t*)j.data(), {j.shape(0), j.shape(1)}); 
}

inline jax_tensor_1 to_jax(ndview<double, 1> arr) {
  size_t shape[1] = {arr.shape[0]};
  return jax_tensor_1(arr.data, 1, shape);
}

inline jax_tensor_2 to_jax(ndview<double, 2> arr) {
  size_t shape[2] = {arr.shape[0], arr.shape[1]};
  return jax_tensor_2(arr.data, 2, shape);
}

inline jax_tensor_3 to_jax(ndview<double, 3> arr) {
  size_t shape[3] = {arr.shape[0], arr.shape[1], arr.shape[2]};
  return jax_tensor_3(arr.data, 3, shape);
}

inline jax_tensor_1u to_jax(ndview<uint64_t, 1> arr) {
  size_t shape[1] = {arr.shape[0]};
  return jax_tensor_1u(arr.data, 1, shape);
}

inline jax_tensor_2u to_jax(ndview<uint64_t, 2> arr) {
  size_t shape[2] = {arr.shape[0], arr.shape[1]};
  return jax_tensor_2u(arr.data, 2, shape);
}

//template < typename tensor_type > 
//tensor_type load(std::string npy_file);
//
//template < typename T, size_t ... shapes > 
//void save(std::string npy_file, nanobind::tensor<nb::jax, T, nb::shape< shapes ... > > arr);