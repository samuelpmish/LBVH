#include "geometry/2D.h"
#include "geometry/morton.h"
#include "geometry/bit_packing.h"

#include "misc/timer.h"
#include "misc/thrust_helper.h"

#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/reduce.h>

#include <algorithm>

namespace Femto2D{

  using hBVH = BVH< MemorySpace::CPU >; 

  template <>
  hBVH::BVH(const hvector< AABB > & leaves) {

    constexpr bool print_timings = false;

    timer stopwatch;

    using vec2f = vec < 2, float >;

    num_leaves = static_cast < int32_t >(leaves.size());

    if constexpr (print_timings) stopwatch.start();
    boxes = hvector < AABB >(2 * num_leaves - 1);
    children = hvector < int2 >(2 * num_leaves - 1);
    ids = hvector < uint32_t >(2 * num_leaves - 1, 0);
    if constexpr (print_timings) stopwatch.stop();
    if constexpr (print_timings) std::cout << "allocation: " << stopwatch.elapsed() << std::endl;

    if constexpr (print_timings) stopwatch.start();
    float big = 1.0e10; 
    global = thrust::reduce(
      leaves.begin(), leaves.end(),
      AABB(big, big, -big, -big),
      [](AABB A, AABB B){ return union_of(A, B); });
    if constexpr (print_timings) stopwatch.stop();
    if constexpr (print_timings) std::cout << "global bounding box: " << stopwatch.elapsed() << std::endl;

    if constexpr (print_timings) stopwatch.start();
    int bits_per_dimension = (sizeof(code_t) * 8) / 2;
    int divisions_per_dimension = 1 << bits_per_dimension;

    vec2f scale = float(divisions_per_dimension - 1) / global.widths();

    hvector< code_t > codes(num_leaves);

    #pragma omp parallel for
    for (int i = 0; i < num_leaves; i++) {
      vec2f u = (leaves[i].center() - global.min()) * scale;

      ids[i] = i;
      codes[i] = morton::encode(static_cast< code_t >(u[0]), 
                                static_cast< code_t >(u[1]));
    }
    if constexpr (print_timings) stopwatch.stop();
    if constexpr (print_timings) std::cout << "calculate morton codes: " << stopwatch.elapsed() << std::endl;

    if constexpr (print_timings) stopwatch.start();
    thrust::stable_sort_by_key(codes.begin(), codes.end(), ids.begin());
    if constexpr (print_timings) stopwatch.stop();
    if constexpr (print_timings) std::cout << "sort morton codes: " << stopwatch.elapsed() << std::endl;

    if constexpr (print_timings) stopwatch.start();
    thrust::gather(ids.begin(), ids.begin() + num_leaves, leaves.begin(), boxes.begin());
    if constexpr (print_timings) stopwatch.stop();
    if constexpr (print_timings) std::cout << "rearrange leaves: " << stopwatch.elapsed() << std::endl;

    if constexpr (print_timings) stopwatch.start();
    hvector < int32_t > parents(2 * num_leaves - 1);
    hvector < int32_t > siblings(2 * num_leaves - 1); 

    parents[num_leaves] = num_leaves;

    #pragma omp parallel for
    for (int32_t i = 0; i < (num_leaves - 1); i++) {

      code_t my_code = codes[i];
      uint32_t my_id = ids[i];

      auto shared_prefix = [&](int other) {
        int common_bits = -1;
        if (0 <= other && other < num_leaves) {
          common_bits = clz(my_code ^ codes[other]);
          if (common_bits == 8 * sizeof(code_t)) {
            common_bits += clz(my_id ^ ids[other]);
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
    if constexpr (print_timings) stopwatch.stop();
    if constexpr (print_timings) std::cout << "radix tree: " << stopwatch.elapsed() << std::endl;

    if constexpr (print_timings) stopwatch.start();
    #pragma omp parallel for
    for (int32_t i = 0; i < num_leaves; i++) {

      // start with the bounding boxes of the leaf nodes
      // and have each thread work its way up the tree
      int32_t id = i;
      int32_t parent = parents[i];
      int32_t sibling = siblings[i];
      AABB box = boxes[i];

      while(id != parent) {

        // only process a parent node if the other
        // sibling has visited the parent as well
        bool first_visit;
        #pragma omp critical 
        {
          ids[parent]++;
          first_visit = ids[parent] == 1;
        }

        if (first_visit) break;

        // compute the union of the two sibling boxes
        AABB sibling_box = boxes[sibling];
        box = union_of(box, sibling_box);

        // move up to the parent node
        id = parent;
        parent = parents[id];
        sibling = siblings[id];

        // and assign the new box to it
        boxes[id] = box;

      }

    }
    if constexpr (print_timings) stopwatch.stop();
    if constexpr (print_timings) std::cout << "bounding boxes: " << stopwatch.elapsed() << std::endl;

  }

  hvector< int2 > hBVH::all_intersections_with(const hvector< vec2 > & points) {

    const int num_points = int(points.size());
    const int num_threads = omp_get_max_threads();
    
    uint32_t hits_per_query = 50;
    uint32_t factor = hits_per_query / num_threads;
    uint32_t capacity = static_cast<uint32_t>(factor * (num_points + num_leaves) / 2);

    auto pairs = std::vector < std::vector < int2 > >(num_threads);
    for (int i = 0; i < num_threads; i++) {
      pairs[i].reserve(capacity);
    }

    #pragma omp parallel for
    for (int i = 0; i < num_points; i++) {

      int tid = omp_get_thread_num();

      int32_t query_id = i;
      vec2 query_point = points[i];

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

        bool overlap_left = intersecting(boxes[left], query_point);
        if (overlap_left && (left < num_leaves)) {
          pairs[tid].push_back(int2{ids[left], query_id});
        }

        bool overlap_right = intersecting(boxes[right], query_point);
        if (overlap_right && (right < num_leaves)) {
          pairs[tid].push_back(int2{ids[right], query_id});
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
      }

      while (n != -1);

    }

    std::vector < size_t > offsets(num_threads+1);
    offsets[0] = 0; 
    for (int i = 0; i < num_threads; i++) {
      offsets[i+1] = offsets[i] + pairs[i].size();
    }

    hvector< int2 > combined;
    combined.resize(offsets[num_threads]);
    #pragma omp parallel for
    for (int i = 0; i < num_threads; i++) {
      std::memcpy(
          &combined[offsets[i]], 
          &(pairs[i][0]), 
          pairs[i].size() * sizeof(int2)
      );
    }

    return combined;

  }

}