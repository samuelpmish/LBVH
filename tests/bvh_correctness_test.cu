#include <gtest/gtest.h>

#include "BVH.hpp"

#include <random>

template < int dim >
using AABB = fm::AABB<dim>;

float random_real() {
  static std::default_random_engine generator;
  static std::uniform_real_distribution<float> distribution(-1.0,1.0);
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

template < int dim >
void box_selfintersection_test(int n, float radius) {

  fm::AABB<dim> global_box;
  for (int i = 0; i < dim; i++) {
    global_box.min[i] = -1.0f;
    global_box.max[i] = +1.0f;
  }

  auto boxes = random_AABBs<dim>(n, radius);

  std::vector< std::array<int, 2> > pairs1;
  for (int i = 0; i < n; i++) {
    for (int j = i+1; j < n; j++) {
      if (intersecting(boxes[i], boxes[j])) {
        pairs1.push_back({i, j});
      }
    }
  }

  GPU::BVH<dim> bvh(boxes, global_box);

  int max_pairs = 1000000;

  int2 * d_pairs2;
  cudaMalloc(&d_pairs2, sizeof(int2) * max_pairs);

  int pairs_found = 0;
  find_intersections(bvh, d_pairs2, max_pairs, pairs_found);

  std::vector< std::array<int, 2> > pairs2(pairs_found);
  cudaMemcpy(&pairs2[0], d_pairs2, sizeof(int2) * pairs_found, cudaMemcpyDeviceToHost);

  cudaFree(d_pairs2);

  std::cout << pairs1.size() << " " << pairs2.size() << std::endl;

  EXPECT_EQ(pairs1.size(), pairs2.size());

  //std::sort(pairs1.begin(), pairs1.end());
  //std::sort(pairs2.begin(), pairs2.end());

  //for (int i = 0; i < pairs1.size(); i++) {
  //  EXPECT_EQ(pairs1[i], pairs2[i]);
  //}

}

template < int dim >
void box_intersection_test(int n, float radius) {

  fm::AABB<dim> global_box;
  for (int i = 0; i < dim; i++) {
    global_box.min[i] = -1.0f;
    global_box.max[i] = +1.0f;
  }

  auto boxes_A = random_AABBs<dim>(n, radius);
  auto boxes_B = random_AABBs<dim>(n, radius);

  std::vector< std::array<int, 2> > pairs1;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (intersecting(boxes_A[i], boxes_B[j])) {
        pairs1.push_back({i, j});
      }
    }
  }

  GPU::BVH<dim> bvh(boxes_A, global_box);

  int max_pairs = 1000000;

  int2 * d_pairs2;
  cudaMalloc(&d_pairs2, sizeof(int2) * max_pairs);

  fm::AABB<dim> * d_boxes_B;
  cudaMalloc(&d_boxes_B, sizeof(fm::AABB<dim>) * n);
  cudaMemcpy(d_boxes_B, &boxes_B[0], sizeof(fm::AABB<dim>) * n, cudaMemcpyHostToDevice);

  int pairs_found = 0;
  find_intersections(bvh, d_boxes_B, n, d_pairs2, max_pairs, pairs_found);

  std::vector< std::array<int, 2> > pairs2(pairs_found);
  cudaMemcpy(&pairs2[0], d_pairs2, sizeof(int2) * pairs_found, cudaMemcpyDeviceToHost);

  cudaFree(d_pairs2);
  cudaFree(d_boxes_B);

  std::cout << pairs1.size() << " " << pairs2.size() << std::endl;

  EXPECT_EQ(pairs1.size(), pairs2.size());

  //std::sort(pairs1.begin(), pairs1.end());
  //std::sort(pairs2.begin(), pairs2.end());

  //for (int i = 0; i < pairs1.size(); i++) {
  //  EXPECT_EQ(pairs1[i], pairs2[i]);
  //}

}

TEST(UnitTest, BVHSelfIntersection2D) {
  box_selfintersection_test<2>(10000, 0.02f);
}

TEST(UnitTest, BVHSelfIntersection3D) {
  box_selfintersection_test<3>(10000, 0.07f);
}

TEST(UnitTest, BVHIntersection2D) {
  box_intersection_test<2>(10000, 0.02f);
}

TEST(UnitTest, BVHIntersection3D) {
  box_intersection_test<3>(10000, 0.07f);
}