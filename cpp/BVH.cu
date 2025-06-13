#include "BVH.hpp"
#include "morton.h"
#include "bit_packing.h"

#include "timer.h"

#include "cuda.h"

#include "util.cuh"
#include "trove/ptr.h"
#include "generics/ldg.h"

__global__ void morton_kernel(
  uint64_t * code_ids,
  const fm::AABB<2> global,
  const fm::AABB<2> * boxes,
  const size_t num_boxes) {

  constexpr int dim = 2;

  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  int b = bits_needed_d(num_boxes);
  int bits_per_dimension = (64 - b) / dim;
  int divisions_per_dimension = 1 << bits_per_dimension;

  float scale = float(divisions_per_dimension - 1);

  fm::vec<dim,float> xy_scale = scale / (global.max - global.min);

  trove::coalesced_ptr < const fm::AABB<dim> > trove_boxes(boxes);

  if (tid < num_boxes) {

    fm::AABB<dim> bv = trove_boxes[tid];

    // get the centroid of the ith bounding box
    fm::vec<dim,float> center = 0.5f * (bv.max + bv.min);

    uint64_t ux = (uint64_t)((center[0] - global.min[0]) * xy_scale[0]);
    uint64_t uy = (uint64_t)((center[1] - global.min[1]) * xy_scale[1]);

    uint64_t code = morton::encode(ux, uy); 

    code_ids[tid] = (code << b) + uint64_t(tid);

  }

}

__global__ void morton_kernel(
  uint64_t * code_ids,
  const fm::AABB<3> global,
  const fm::AABB<3> * boxes,
  const size_t num_boxes) {

  constexpr int dim = 3;

  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  int b = bits_needed_d(num_boxes);
  int bits_per_dimension = (64 - b) / dim;
  int divisions_per_dimension = 1 << bits_per_dimension;

  float scale = float(divisions_per_dimension - 1);

  fm::vec<dim,float> xyz_scale = scale / (global.max - global.min);

  trove::coalesced_ptr < const fm::AABB<dim> > trove_boxes(boxes);

  if (tid < num_boxes) {

    fm::AABB<dim> bv = trove_boxes[tid];

    // get the centroid of the ith bounding box
    fm::vec<dim,float> center = 0.5f * (bv.max + bv.min);

    uint64_t ux = (uint64_t)((center[0] - global.min[0]) * xyz_scale[0]);
    uint64_t uy = (uint64_t)((center[1] - global.min[1]) * xyz_scale[1]);
    uint64_t uz = (uint64_t)((center[2] - global.min[2]) * xyz_scale[2]);

    uint64_t code = morton::encode(ux, uy, uz); 

    code_ids[tid] = (code << b) + uint64_t(tid);

  }

}

template < typename T >
__global__ void permute_objects(
        T * permuted_objects,
  const T * unsorted_objects,
  const uint64_t * codes,
  const int num_objects) {

  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  trove::coalesced_ptr < T > trove_objects(permuted_objects);

  uint64_t mask = (uint64_t(1) << bits_needed_d(num_objects)) - 1;

  if (tid < num_objects) {
    uint64_t id = codes[tid] & mask;
    trove_objects[tid] = __ldg(&unsorted_objects[id]);
  }

}

class PrefixComparator{

  public:

    const uint64_t   base;
    const uint64_t * codes;
    const int        n;

    __host__ __device__
    PrefixComparator(const uint64_t    _base,
                     const uint64_t  * _codes,
                     const int         _n) :
      base(_base), codes(_codes), n(_n){}

    __device__ __forceinline__
    int operator()(int i){
      return (i >= 0 && i < n) ? __clzll(base ^ codes[i]) : -1;
    }

};

__global__ void connect_radix_tree(
     int32_t * parents,
        int2 * children,
        int2 * rightmost_leaf_in_subtree,
    uint64_t * codes,
     int32_t   n) {

  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid < (n-1)){

    PrefixComparator shared_prefix(codes[tid], codes, n);

    // Choose search direction.
    int prefix_prev = shared_prefix(tid-1);
    int prefix_next = shared_prefix(tid+1);
    int prefix_min = min(prefix_prev, prefix_next);

    int d = (prefix_next > prefix_prev) ? 1 : -1;

    // Find upper bound for length.
    int lmax = 32;
    uint32_t probe;
    do{
      lmax <<= 2;
      probe = tid + lmax * d;
    } while (probe < n && shared_prefix(probe) > prefix_min);

    // Determine length.
    int l = 0;
    for(int t = lmax >> 1; t > 0; t >>= 1){
      probe = tid + (l + t) * d;
      if (probe < n & shared_prefix(probe) > prefix_min) {
        l += t;
      }
    }
    int j = tid + l * d;
    int prefix_node = shared_prefix(j);

    // Find split point.
    int s = 0;
    int t = l;
    do{
      t = (t + 1) >> 1;
      probe = tid + (s + t) * d;
      if (probe < n && shared_prefix(probe) > prefix_node) {
        s += t;
      }
    } while (t > 1);
    int k = tid + s * d + ::min(d, 0);

    // Output node.
    int32_t lo = ::min(tid, j);
    int32_t hi = ::max(tid, j);

    int32_t left  = (lo == k    ) ? k + 0 : (k + 0 + n);
    int32_t right = (hi == k + 1) ? k + 1 : (k + 1 + n);

    parents[left] = tid + n;
    parents[right] = tid + n;
    children[tid + n] = int2{left, right};
    rightmost_leaf_in_subtree[tid + n] = int2{k, hi};

    if(tid == 0){
      parents[n] = n;
    }

  }
}

template < int dim >
__global__ void update_tree_aabbs(
    fm::AABB<dim> * boxes,
    int * visited,
    const int32_t * parents,
    const    int2 * children,
    const int32_t n){

  trove::coalesced_ptr < fm::AABB<dim> > trove_boxes(boxes);

  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid < n){

    // start with the bounds of the leaf nodes
    fm::AABB<dim> bv = trove_boxes[tid];
    int32_t parent = parents[tid];

    // check if the parent's bounds have already been set
    while(atomicAdd(&visited[parent], 1) == 1) {

      // if the parent node hasn't been processed yet,
      // set its bounding volume equal to the union of
      // its children's bounding volumes
      int2 c = children[parent];
      int sibling = (tid == c.x) ? c.y : c.x;

      fm::AABB<dim> sibling_bv = trove_boxes[sibling];

      bv = union_of(bv, sibling_bv);

      tid = parent;
      parent = parents[tid];

      boxes[tid] = bv;

    }

  }

}

template < int dim >
__global__ void self_traverse(
    int2 * pairs,
    int * num_pairs, 
    const fm::AABB<dim> * boxes,
    const uint64_t * code_ids,
    const int2 * children,
    const int2 * rightmost_leaf_in_subtree,
    uint64_t num_leaves) {

  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  uint64_t mask = (uint64_t(1) << bits_needed_d(num_leaves)) - 1;

  if (tid < num_leaves) {

    int32_t query_id = code_ids[tid] & mask;
    fm::AABB<dim> query_box = __ldg(&boxes[tid]);
  
    // Allocate traversal stack from thread-local memory,
    // and push NULL to indicate that there are no postponed nodes.
    int   stack[32];
    int * stack_ptr = stack;
    *stack_ptr++ = NULL; // push
  
    // Traverse nodes starting from the root.
  
    int node = num_leaves;
    do {

      // Check each child node for overlap.
      int2 c = __ldg(&children[node]);
      int child_left  = c.x;
      int child_right = c.y;
      
      int2 rightmost_subtree_leaf = rightmost_leaf_in_subtree[node];

      bool overlap_left  = intersecting(query_box, __ldg(&boxes[child_left]));
      bool overlap_right = intersecting(query_box, __ldg(&boxes[child_right]));

      if (rightmost_subtree_leaf.x <= tid) overlap_left = false;
      if (rightmost_subtree_leaf.y <= tid) overlap_right = false;

      // If the query overlaps with a leaf node, report a collision.
      if (overlap_left && (child_left < num_leaves)) {

        int pair_id = atomicAdd(num_pairs, 1);
        int child_left_id = code_ids[child_left] & mask;

        if (query_id < child_left_id) {
          pairs[pair_id] = int2{query_id, child_left_id};
        } else {
          pairs[pair_id] = int2{child_left_id, query_id};
        }

      }

      if (overlap_right && (child_right < num_leaves)) {

        int pair_id = atomicAdd(num_pairs, 1);
        int child_right_id = code_ids[child_right] & mask;

        if (query_id < child_right_id) {
          pairs[pair_id] = int2{query_id, child_right_id};
        } else {
          pairs[pair_id] = int2{child_right_id, query_id};
        }

      }

      // traverse when a query overlaps with an internal node
      bool traverse_left  = (overlap_left  && (child_left  >= num_leaves));
      bool traverse_right = (overlap_right && (child_right >= num_leaves));

      if (!traverse_left && !traverse_right) {
        node = *--stack_ptr; // pop
      } else {
        node = (traverse_left) ? child_left : child_right;
        if (traverse_left && traverse_right) {
          *stack_ptr++ = child_right; // push
        }
      }
    }

    while (node != NULL);

  }

}

template < int dim >
__global__ void traverse(
    int2 * pairs,
    int * num_pairs, 
    const fm::AABB<dim> * query_boxes,
    const fm::AABB<dim> * bvh_boxes,
    const uint64_t * bvh_code_ids,
    const int2 * bvh_children,
    uint64_t num_leaves,
    uint64_t num_queries) {

  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  uint64_t bvh_mask = (uint64_t(1) << bits_needed_d(num_leaves)) - 1;
  uint64_t query_mask = (uint64_t(1) << bits_needed_d(num_queries)) - 1;

  if (tid < num_queries) {

    fm::AABB<dim> query_box = __ldg(&query_boxes[tid]);
  
    // Allocate traversal stack from thread-local memory,
    // and push NULL to indicate that there are no postponed nodes.
    int   stack[32];
    int * stack_ptr = stack;
    *stack_ptr++ = NULL; // push
  
    // Traverse nodes starting from the root.
    int node = num_leaves;
    do {

      // Check each child node for overlap.
      int2 c = __ldg(&bvh_children[node]);
      int child_left  = c.x;
      int child_right = c.y;

      bool overlap_left  = intersecting(__ldg(&bvh_boxes[child_left]), query_box);
      bool overlap_right = intersecting(__ldg(&bvh_boxes[child_right]), query_box);

      // If the query overlaps with a leaf node, report a collision.
      if (overlap_left && (child_left < num_leaves)) {

        int pair_id = atomicAdd(num_pairs, 1);
        int child_left_id = __ldg(&bvh_code_ids[child_left]) & bvh_mask;
        pairs[pair_id] = int2{child_left_id, tid};

      }

      if (overlap_right && (child_right < num_leaves)) {

        int pair_id = atomicAdd(num_pairs, 1);
        int child_right_id = __ldg(&bvh_code_ids[child_right]) & bvh_mask;
        pairs[pair_id] = int2{child_right_id, tid};

      }

      // traverse when a query overlaps with an internal node
      bool traverse_left  = (overlap_left  && (child_left  >= num_leaves));
      bool traverse_right = (overlap_right && (child_right >= num_leaves));

      if (!traverse_left && !traverse_right) {
        node = *--stack_ptr; // pop
      } else {
        node = (traverse_left) ? child_left : child_right;
        if (traverse_left && traverse_right) {
          *stack_ptr++ = child_right; // push
        }
      }
    }

    while (node != NULL);

  }

}

namespace GPU {

template < int dim >
void morton_sort(fm::AABB<dim> global, uint64_t * sorted_codes, fm::AABB<dim> * sorted_objects, const fm::AABB<dim> * unsorted_objects, int num_objects) {

  int blocksize = 128; 
  int gridsize = (num_objects + blocksize - 1) / blocksize;

  uint64_t * unsorted_codes;
  cudaMalloc(&unsorted_codes, sizeof(uint64_t) * num_objects);
  morton_kernel<<< gridsize, blocksize >>>(
      unsorted_codes, 
      global, 
      unsorted_objects, 
      num_objects);

  void * sort_buffer = NULL;
  size_t bytes = 0;

  // first invocation returns the number of bytes needed
  util::sort(sort_buffer, bytes, unsorted_codes, sorted_codes, num_objects);
  cudaMalloc(&sort_buffer, bytes);

  // sceond invocation actually performs the sort
  util::sort(sort_buffer, bytes, unsorted_codes, sorted_codes, num_objects);

  permute_objects<<< gridsize, blocksize >>>(sorted_objects, unsorted_objects, sorted_codes, num_objects);

  cudaFree(sort_buffer);
  cudaFree(unsorted_codes);

}

template < int dim >
BVH<dim>::BVH(const std::vector< fm::AABB<dim> > & h_boxes, fm::AABB<dim> global_box) {

  global = global_box;
  num_leaves = h_boxes.size();

  // member variables
  code_ids.resize(num_leaves);
  boxes.resize(2 * num_leaves);
  children.resize(2 * num_leaves);
  rightmost_leaf_in_subtree.resize(2 * num_leaves);

  // temporary storage 
  fm::AABB<dim> * d_boxes;
  cudaMalloc(&d_boxes, sizeof(fm::AABB<dim>) * num_leaves);

  cudaMemcpy(d_boxes, &h_boxes[0], sizeof(fm::AABB<dim>) * num_leaves, cudaMemcpyHostToDevice); 

  // calculate the morton codes for each box and sort them by that index
  morton_sort(global_box, thrust::raw_pointer_cast(code_ids.data()), thrust::raw_pointer_cast(boxes.data()), d_boxes, num_leaves);

  int * parents;
  cudaMalloc(&parents, sizeof(int) * (2 * num_leaves));

  int * visited;
  cudaMalloc(&visited, sizeof(int) * (2 * num_leaves));

  uint64_t mask = (uint64_t(1) << bits_needed(num_leaves)) - 1;

  // determine the parent-child connectivity for the radix tree
  {
    int blocksize = 128; 
    int gridsize = (num_leaves + blocksize - 2) / blocksize;

    connect_radix_tree<<< gridsize, blocksize >>>(
        parents, 
        thrust::raw_pointer_cast(children.data()), 
        thrust::raw_pointer_cast(rightmost_leaf_in_subtree.data()), 
        thrust::raw_pointer_cast(code_ids.data()), 
        num_leaves);
  }

  // assign tight bounding boxes around the internal nodes of the BVH
  {
    int blocksize = 128; 
    int gridsize = (num_leaves + blocksize - 2) / blocksize;

    cudaMemset(visited, 0, sizeof(int) * 2 * num_leaves);
    update_tree_aabbs<<< gridsize, blocksize >>>(
        thrust::raw_pointer_cast(boxes.data()),
        visited,
        parents,
        thrust::raw_pointer_cast(children.data()),
        num_leaves);
  }

  cudaFree(d_boxes);
  cudaFree(parents);
  cudaFree(visited);

}

template BVH< 2 >::BVH(const std::vector < fm::AABB<2> > &, fm::AABB<2>);
template BVH< 3 >::BVH(const std::vector < fm::AABB<3> > &, fm::AABB<3>);

template < int dim >
void find_intersections(const BVH<dim> & bvh, int2 * intersecting_pairs, int max_pairs, int & pairs_found) {

  int blocksize = 256;
  int gridsize = (bvh.num_leaves + blocksize - 1) / blocksize;

  int * num_pairs;
  cudaMalloc(&num_pairs, sizeof(int));
  cudaMemset(num_pairs, 0, sizeof(int));

  self_traverse<<<gridsize, blocksize>>>(intersecting_pairs, 
                                         num_pairs, 
                                         thrust::raw_pointer_cast(bvh.boxes.data()),
                                         thrust::raw_pointer_cast(bvh.code_ids.data()),
                                         thrust::raw_pointer_cast(bvh.children.data()),
                                         thrust::raw_pointer_cast(bvh.rightmost_leaf_in_subtree.data()),
                                         bvh.num_leaves);

  cudaMemcpy(&pairs_found, num_pairs, sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(num_pairs);

}

template void find_intersections(const BVH<2> &, int2 *, int, int &);
template void find_intersections(const BVH<3> &, int2 *, int, int &);

template < int dim >
void find_intersections(const BVH<dim> & bvh, const fm::AABB<dim> * query_boxes, int num_query_boxes, int2 * intersecting_pairs, int max_pairs, int & pairs_found) {

  int blocksize = 256;
  int gridsize = (num_query_boxes + blocksize - 1) / blocksize;

  int * num_pairs;
  cudaMalloc(&num_pairs, sizeof(int));
  cudaMemset(num_pairs, 0, sizeof(int));

  traverse<<<gridsize, blocksize>>>(intersecting_pairs, 
                                    num_pairs, 
                                    query_boxes,
                                    thrust::raw_pointer_cast(bvh.boxes.data()),
                                    thrust::raw_pointer_cast(bvh.code_ids.data()),
                                    thrust::raw_pointer_cast(bvh.children.data()),
                                    bvh.num_leaves,
                                    num_query_boxes);

  cudaMemcpy(&pairs_found, num_pairs, sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(num_pairs);

}

template void find_intersections(const BVH<2> &, const fm::AABB<2> *, int, int2 *, int, int &);
template void find_intersections(const BVH<3> &, const fm::AABB<3> *, int, int2 *, int, int &);


#if 0
template < typename T >
array < int2, GPU_MEMORY > bvh< GPU_MEMORY >::intersect(
    const array < T, GPU_MEMORY > & unsorted_query_objects) {

  timer stopwatch;

  stopwatch.start();
  int num_queries = unsorted_query_objects.size();
  int hits_per_query = 50;
  uint64_t capacity = hits_per_query * (num_queries + num_leaves) / 2;
  array < int2, GPU_MEMORY > pairs(capacity);
  array < T, GPU_MEMORY > query_objects(num_queries);
  array < uint64_t, GPU_MEMORY > query_code_ids(num_queries);
  stopwatch.stop();
  std::cout << "preallocation: " << stopwatch.elapsed() << "s" << std::endl;

  stopwatch.start();
  morton_sort< T >(query_code_ids, query_objects, unsorted_query_objects);
  cudaDeviceSynchronize();
  stopwatch.stop();
  std::cout << "morton sort: " << stopwatch.elapsed() << "s" << std::endl;

  stopwatch.start();
  int * num_pairs;
  cudaMalloc(&num_pairs, sizeof(int));
  cudaMemset(num_pairs, 0, sizeof(int));

  int blocksize = 128; 
  int gridsize = (num_queries + blocksize - 1) / blocksize;
  traverse< T ><<< gridsize, blocksize >>>(
      pairs,
      num_pairs,
      query_objects,
      query_code_ids,
      boxes,
      code_ids,
      children,
      parents,
      num_leaves,
      num_queries,
      flipped);
  cudaDeviceSynchronize();
  stopwatch.stop();
  std::cout << "traverse: " << stopwatch.elapsed() << "s" << std::endl;

  stopwatch.start();
  int num_pairs_h;
  cudaMemcpy(&num_pairs_h, num_pairs, sizeof(int), cudaMemcpyDefault);
  cudaFree(num_pairs);

  pairs.shrink(num_pairs_h);

  return pairs;

}

template array < int2, GPU_MEMORY > bvh< GPU_MEMORY >::intersect(const array < aabb, GPU_MEMORY > &);
template array < int2, GPU_MEMORY > bvh< GPU_MEMORY >::intersect(const array < vec3, GPU_MEMORY > &);
#endif

}