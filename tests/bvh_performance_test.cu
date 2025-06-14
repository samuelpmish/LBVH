#include <gtest/gtest.h>

#include "BVH.hpp"

#include "binary_io.hpp"

#include <random>

template < int dim >
using AABB = fm::AABB<dim>;

template < typename T >
std::vector<T> host_copy(const thrust::device_vector<T> & d_buffer) {
  std::vector<T> h_buffer(d_buffer.size());
  cudaMemcpy(&h_buffer[0], thrust::raw_pointer_cast(d_buffer.data()), sizeof(T) * d_buffer.size(), cudaMemcpyDeviceToHost);
  return h_buffer;
}

float random_real() {
  static std::default_random_engine generator;
  static std::uniform_real_distribution<float> distribution(-1.0, 1.0);
  return distribution(generator);
};

template < int dim >
std::vector< AABB<dim> > random_AABBs(int n, float radius) {
  std::vector< AABB<dim> > boxes(n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < dim; j++) {
      float center = random_real();
      boxes[i].min[j] = center - radius;
      boxes[i].max[j] = center + radius;
    }
  }
  return boxes;
}

static constexpr int num_iter = 5;

template < int dim >
void box_selfintersection_perftest(int n, float radius) {

  fm::AABB<dim> global_box;
  for (int i = 0; i < dim; i++) {
    global_box.min[i] = -1.0f;
    global_box.max[i] = +1.0f;
  }

  auto boxes = random_AABBs<dim>(n, radius);

  int max_pairs = 10000000;

  int2 * d_pairs2;
  cudaMalloc(&d_pairs2, sizeof(int2) * max_pairs);

  fm::AABB<dim> * d_boxes;
  cudaMalloc(&d_boxes, sizeof(fm::AABB<dim>) * n);
  cudaMemcpy(d_boxes, &boxes[0], sizeof(fm::AABB<dim>) * n, cudaMemcpyHostToDevice);

  for (int k = 0; k < num_iter; k++) {

    GPU::BVH<dim> bvh(boxes, global_box);

    int pairs_found = 0;
    find_intersections(bvh, d_pairs2, max_pairs, pairs_found);

    std::cout << pairs_found << std::endl;

  }

  cudaFree(d_pairs2);
  cudaFree(d_boxes);

}

template < int dim >
void box_intersection_perftest(int n, float radius) {

  fm::AABB<dim> global_box;
  for (int i = 0; i < dim; i++) {
    global_box.min[i] = -1.0f;
    global_box.max[i] = +1.0f;
  }

  auto boxes_A = random_AABBs<dim>(n, radius);
  auto boxes_B = random_AABBs<dim>(n, radius);

  int max_pairs = 10000000;
  int2 * d_pairs2;
  cudaMalloc(&d_pairs2, sizeof(int2) * max_pairs);

  fm::AABB<dim> * d_boxes_B;
  cudaMalloc(&d_boxes_B, sizeof(fm::AABB<dim>) * n);
  cudaMemcpy(d_boxes_B, &boxes_B[0], sizeof(fm::AABB<dim>) * n, cudaMemcpyHostToDevice);

  for (int k = 0; k < 5; k++) {

    GPU::BVH<dim> bvh(boxes_A, global_box);

    int pairs_found = 0;
    find_intersections(bvh, d_boxes_B, n, d_pairs2, max_pairs, pairs_found);

    std::cout << pairs_found << std::endl;

  }

  cudaFree(d_pairs2);
  cudaFree(d_boxes_B);

}

TEST(PerfTest, BVHSelfIntersection2D) {
  box_selfintersection_perftest<2>(10000, 0.1f);
  box_selfintersection_perftest<2>(100000, 0.01f);
  box_selfintersection_perftest<2>(1000000, 0.001f);
}

TEST(PerfTest, BVHSelfIntersection3D) {
  box_selfintersection_perftest<3>(10000, 0.07f);
  box_selfintersection_perftest<3>(100000, 0.03f);
  box_selfintersection_perftest<3>(1000000, 0.007f);
}

TEST(PerfTest, BVHIntersection2D) {
  box_intersection_perftest<2>(10000, 0.1f);
  box_intersection_perftest<2>(100000, 0.01f);
  box_intersection_perftest<2>(1000000, 0.001f);
}

TEST(PerfTest, BVHIntersection3D) {
  box_intersection_perftest<3>(10000, 0.12f);
  box_intersection_perftest<3>(100000, 0.03f);
  box_intersection_perftest<3>(1000000, 0.007f);
}