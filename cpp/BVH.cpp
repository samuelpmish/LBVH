#include "BVH.hpp"
#include "morton.hpp"

#include <tuple>
#include <algorithm>

#include <iostream>

template < int dim >
BVH<dim>::BVH(const std::vector< AABB_t > & leaves) {

  num_leaves = static_cast < int32_t >(leaves.size());

  if (num_leaves == 0) return;

  boxes = std::vector < AABB_t >(2 * num_leaves - 1);
  children = std::vector < int2 >(2 * num_leaves - 1);
  ids = std::vector < int32_t >(2 * num_leaves - 1, 0);

  global = leaves[0];

  for (int i = 0; i < num_leaves; i++) {
    AABB_t leaf = leaves[i];
    for (int d = 0; d < dim; d++) {
      global.min[d] = std::min(global.min[d], leaf.min[d]);
      global.max[d] = std::max(global.max[d], leaf.max[d]);
    }
  }

  float widths[dim];
  for (int d = 0; d < dim; d++) {
    widths[d] = global.max[d] - global.min[d];
  }

  std::vector < std::tuple< code_t, uint32_t > > code_ids(num_leaves);

  for (int i = 0; i < num_leaves; i++) {

    // scaled coordinates in the range (0, 1) 
    float u[dim];
    for (int d = 0; d < dim; d++) {
      u[d] = (0.5f * (leaves[i].min[d] + leaves[i].max[d]) - global.min[d]) / widths[d];
    }

    code_ids[i] = {morton::encode(u), uint32_t(i)};

  }

  std::sort(code_ids.begin(), code_ids.end());

  for (int i = 0; i < num_leaves; i++) {
    auto id = std::get<1>(code_ids[i]);
    boxes[i] = leaves[id];
    ids[i] = id;
  }

  std::vector < int32_t > parents(2 * num_leaves - 1);
  std::vector < int32_t > siblings(2 * num_leaves - 1); 

  parents[num_leaves] = num_leaves;

  for (int32_t i = 0; i < (num_leaves - 1); i++) {

    uint64_t my_code = std::get<0>(code_ids[i]);
    uint32_t my_id = std::get<1>(code_ids[i]);

    auto shared_prefix = [&](int other) {
      int common_bits = -1;
      if (0 <= other && other < num_leaves) {
        common_bits = clz(my_code ^ std::get<0>(code_ids[other]));
        if (common_bits == 8 * sizeof(code_t)) {
          common_bits += clz(my_id ^ std::get<1>(code_ids[other]));
        }
      }
      return common_bits;
    };

    // Choose search direction.
    int32_t prefix_prev = shared_prefix(i-1);
    int32_t prefix_next = shared_prefix(i+1);
    int32_t prefix_min = std::min(prefix_prev, prefix_next);

    int32_t d = (prefix_next > prefix_prev) ? 1 : -1;

    // Find upper bound for length.
    int32_t lmax = 32;
    int32_t probe;
    do{
      lmax <<= 2;
      probe = i + lmax * d;
    } while (probe < num_leaves && shared_prefix(probe) > prefix_min);

    // Determine length.
    int32_t l = 0;
    for(int32_t t = lmax >> 1; t > 0; t >>= 1){
      probe = i + (l + t) * d;
      if ((probe < num_leaves) & (shared_prefix(probe) > prefix_min)) {
        l += t;
      }
    }
    int32_t j = i + l * d;
    int32_t prefix_node = shared_prefix(j);

    // Find split point.
    int32_t s = 0;
    int32_t t = l;
    do{
      t = (t + 1) >> 1;
      probe = i + (s + t) * d;
      if (probe < num_leaves && shared_prefix(probe) > prefix_node) {
        s += t;
      }
    } while (t > 1);
    int32_t k = i + s * d + std::min(d, 0);

    // Output node.
    int32_t lo = std::min(i, j);
    int32_t hi = std::max(i, j);

    int32_t left  = (lo == k    ) ? k + 0 : (k + 0 + num_leaves);
    int32_t right = (hi == k + 1) ? k + 1 : (k + 1 + num_leaves);

    parents[left] = i + num_leaves;
    parents[right] = i + num_leaves;
    siblings[left] = right;
    siblings[right] = left;

    children[i + num_leaves] = int2{left, right};
  }

  for (int32_t i = 0; i < num_leaves; i++) {

    // start with the bounding boxes of the leaf nodes
    // and have each thread work its way up the tree
    int32_t id = i;
    int32_t parent = parents[i];
    int32_t sibling = siblings[i];
    AABB_t box = boxes[i];

    while(id != parent) {

      // only process a parent node if the other
      // sibling has visited the parent as well
      bool first_visit;
      {
        ids[parent]++;
        first_visit = ids[parent] == 1;
      }

      if (first_visit) break;

      // compute the union of the two sibling boxes
      AABB_t sibling_box = boxes[sibling];
      box = bounding_box(box, sibling_box);

      // move up to the parent node
      id = parent;
      parent = parents[id];
      sibling = siblings[id];

      // and assign the new box to it
      boxes[id] = box;

    }

  }

}

template struct BVH<2>;
template struct BVH<3>;