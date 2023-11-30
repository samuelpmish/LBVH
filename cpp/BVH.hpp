#pragma once

#include <vector>
#include <cstdint>

#include "AABB.hpp"

template < int dim >
struct BVH {
  struct int2 { 
    int x;
    int y; 
  };

  using code_t = uint64_t;
  using AABB_t = AABB<dim,float>;

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
  void query(AABB_t box, callable f) {

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