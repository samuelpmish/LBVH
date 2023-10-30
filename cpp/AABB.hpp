#pragma once

#include "tensor.hpp"

#include <algorithm>

template < int dim, typename T = float >
struct AABB { 
  tensor<T, dim> min; 
  tensor<T, dim> max; 

  T SDF(const tensor<T,dim> & p) const {
    constexpr T zero{};
    auto center = (min + max) * 0.5f;
    auto halfwidths = (max - min) * 0.5f;
    auto q = abs(p - center) - halfwidths;
    return norm(::max(q,zero)) + std::min(std::max(q[0],std::max(q[1], q[2])), zero);
  }
};

template< typename T = float >
AABB<2, T> bounding_box(AABB<2, T> a, AABB<2, T> b) {
  return AABB<2, T>{
    {std::min(a.min[0], b.min[0]), std::min(a.min[1], b.min[1])},
    {std::max(a.max[0], b.max[0]), std::max(a.max[1], b.max[1])}
  };
}

template< typename T = float >
AABB<3, T> bounding_box(AABB<3, T> a, AABB<3, T> b) {
  return AABB<3, T>{
    {std::min(a.min[0], b.min[0]), std::min(a.min[1], b.min[1]), std::min(a.min[2], b.min[2])},
    {std::max(a.max[0], b.max[0]), std::max(a.max[1], b.max[1]), std::max(a.max[2], b.max[2])}
  };
}

template < typename T >
bool intersecting(AABB<2,T> a, AABB<2,T> b) {
  return a.min[0] <= b.max[0] && a.min[1] <= b.max[1] && a.max[0] >= b.min[0] && a.max[1] >= b.min[1];
}

template < typename T >
bool intersecting(AABB<3,T> a, AABB<3,T> b) {
  return a.min[0] <= b.max[0] && a.min[1] <= b.max[1] && a.min[2] <= b.max[2] && 
         a.max[0] >= b.min[0] && a.max[1] >= b.min[1] && a.max[2] >= b.min[2];
}

template < typename T >
AABB<2,T> intersection_of(AABB<2,T> a, AABB<2,T> b) {
  return AABB<2,T>{
    {std::max(a.min[0], b.min[0]), std::max(a.min[1], b.min[1])},
    {std::min(a.max[0], b.max[0]), std::min(a.max[1], b.max[1])}
  };
}

template < typename T >
AABB<3,T> intersection_of(AABB<3,T> a, AABB<3,T> b) {
  return AABB<3,T>{
    {std::max(a.min[0], b.min[0]), std::max(a.min[1], b.min[1]), std::max(a.min[2], b.min[2])},
    {std::min(a.max[0], b.max[0]), std::min(a.max[1], b.max[1]), std::min(a.max[2], b.max[2])},
  };
}