#pragma once

#include <vector>
#include <cstdint>

#include "fm/types/AABB.hpp"

template < int dim >
struct BVH {
  struct int2 { 
    int x;
    int y; 
  };

  using code_t = uint64_t;
  using AABB_t = fm::AABB<dim,float>;

  AABB_t global;

  int32_t num_leaves;
  std::vector< int32_t > ids;
  std::vector< AABB_t > boxes;
  std::vector< int2 > children;

  BVH() {}

  BVH(const std::vector< AABB_t > &);

  // traverse the BVH for intersections with `box` 
  // and invoke `f` when finding a hit with a leaf node
  // note: f will be invoked with the argument of the leaf node
  template < typename callable >
  void query(AABB_t box, callable f) const {

    // Allocate traversal stack from thread-local memory,
    // and push NULL to indicate that there are no postponed nodes.
    int   stack[32];
    int * stack_ptr = stack;
    *stack_ptr++ = -1;

    // Traverse nodes starting from the root.
    int n = num_leaves;
    do {

      // Check each child node for overlap.
      auto [left, right] = children[n];

      bool overlap_left = intersecting(boxes[left], box);
      if (overlap_left && (left < num_leaves)) {
        f(ids[left]);
      }

      bool overlap_right = intersecting(boxes[right], box);
      if (overlap_right && (right < num_leaves)) {
        f(ids[right]);
      }

      // traverse when a query overlaps with an internal node
      bool traverse_left  = (overlap_left  && (left  >= num_leaves));
      bool traverse_right = (overlap_right && (right >= num_leaves));

      if (!traverse_left && !traverse_right) {
        n = *--stack_ptr; // pop
      } else {
        n = (traverse_left) ? left : right;
        if (traverse_left && traverse_right) {
          *stack_ptr++ = right; // push
        }
      }
    } while (n != -1);

  }

};

#ifdef LBVH_ENABLE_CUDA

#include "thrust/device_vector.h"

namespace GPU {

template < int dim >
struct BVH {

  using code_t = uint64_t;
  using AABB_t = fm::AABB<dim,float>;

  AABB_t global;

  int32_t num_leaves;
  thrust::device_vector< uint64_t > code_ids;
  thrust::device_vector< AABB_t > boxes;
  thrust::device_vector< int2 > children;
  thrust::device_vector< int2 > rightmost_leaf_in_subtree;

  float time_morton_code;
  float time_sort;
  float time_permute;
  float time_tree_connectivity;
  float time_tree_bounding_boxes; 

  BVH() {}

  BVH(const std::vector< AABB_t > &, AABB_t global);

  #if 0
  // traverse the BVH for intersections with `box` 
  // and invoke `f` when finding a hit with a leaf node
  // note: f will be invoked with the argument of the leaf node
  template < typename callable >
  void query(AABB_t box, callable f) const {

    // Allocate traversal stack from thread-local memory,
    // and push NULL to indicate that there are no postponed nodes.
    int   stack[32];
    int * stack_ptr = stack;
    *stack_ptr++ = -1;

    // Traverse nodes starting from the root.
    int n = num_leaves;
    do {

      // Check each child node for overlap.
      auto [left, right] = children[n];

      bool overlap_left = intersecting(boxes[left], box);
      if (overlap_left && (left < num_leaves)) {
        f(ids[left]);
      }

      bool overlap_right = intersecting(boxes[right], box);
      if (overlap_right && (right < num_leaves)) {
        f(ids[right]);
      }

      // traverse when a query overlaps with an internal node
      bool traverse_left  = (overlap_left  && (left  >= num_leaves));
      bool traverse_right = (overlap_right && (right >= num_leaves));

      if (!traverse_left && !traverse_right) {
        n = *--stack_ptr; // pop
      } else {
        n = (traverse_left) ? left : right;
        if (traverse_left && traverse_right) {
          *stack_ptr++ = right; // push
        }
      }
    } while (n != -1);

  }
  #endif

};

// note: intersecting_pairs is a pointer to device memory
template < int dim >
void find_intersections(const BVH<dim> & bvh, int2 * intersecting_pairs, int max_pairs, int & pairs_found);

template < int dim >
void find_intersections(const BVH<dim> & bvh, const fm::AABB<dim> * query_boxes, int num_query_boxes, int2 * intersecting_pairs, int max_pairs, int & pairs_found);

template < int dim >
struct BVH_view {

  using code_t = uint64_t;
  using AABB_t = fm::AABB<dim,float>;

  int32_t num_leaves;
  const uint64_t * code_ids;
  const AABB_t * boxes;
  const int2 * children;
  const int2 * ranges;

  BVH_view(const BVH<dim> & bvh) {
    num_leaves = bvh.num_leaves;
    code_ids = thrust::raw_pointer_cast(bvh.code_ids.data());
    boxes = thrust::raw_pointer_cast(bvh.boxes.data());
    children = thrust::raw_pointer_cast(bvh.children.data());
  }

#if 0
  // traverse the BVH for intersections with `box` 
  // and invoke `f` when finding a hit with a leaf node
  // note: f will be invoked with the argument of the leaf node
  template < typename callable >
  __device__ void query(AABB_t box, callable f) const {

    // Allocate traversal stack from thread-local memory,
    // and push NULL to indicate that there are no postponed nodes.
    int   stack[32];
    int * stack_ptr = stack;
    *stack_ptr++ = -1;

    // Traverse nodes starting from the root.
    int n = num_leaves;
    do {

      // Check each child node for overlap.
      auto [left, right] = children[n];

      bool overlap_left = intersecting(boxes[left], box);
      if (overlap_left && (left < num_leaves)) {
        f(ids[left]);
      }

      bool overlap_right = intersecting(boxes[right], box);
      if (overlap_right && (right < num_leaves)) {
        f(ids[right]);
      }

      // traverse when a query overlaps with an internal node
      bool traverse_left  = (overlap_left  && (left  >= num_leaves));
      bool traverse_right = (overlap_right && (right >= num_leaves));

      if (!traverse_left && !traverse_right) {
        n = *--stack_ptr; // pop
      } else {
        n = (traverse_left) ? left : right;
        if (traverse_left && traverse_right) {
          *stack_ptr++ = right; // push
        }
      }
    } while (n != -1);

  }
#endif

};

}

#endif