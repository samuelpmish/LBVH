#include <gtest/gtest.h>

#include "BVH.hpp"

#include <random>

template < int dim >
using AABB = fm::AABB<dim>;

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

template < int dim >
void box_selfintersection_test(int n, float radius) {

  fm::AABB<dim> global_box;
  for (int i = 0; i < dim; i++) {
    global_box.min[i] = -1.0f;
    global_box.max[i] = +1.0f;
  }

  auto boxes = random_AABBs<dim>(n, radius);

  std::vector< std::array<int, 2> > pairs1;
  BVH<dim> cpu_bvh(boxes);
  for (int j = 0; j < n; j++) {
    cpu_bvh.query(boxes[j], [&](int i){
      if (i < j) {
        pairs1.push_back({i, j});
      }
    });
  }

  std::sort(pairs1.begin(), pairs1.end());

  std::cout << pairs1.size() << std::endl;

  int max_pairs = 1000000;
  GPU::intersection_list intersections(max_pairs);

  for (int k = 0; k < 5; k++) {

    GPU::BVH<dim> bvh(boxes, global_box);

    find_intersections(intersections, bvh);

    EXPECT_EQ(pairs1.size(), intersections.pairs_found);
    if (pairs1.size() == intersections.pairs_found && intersections.pairs_found <= max_pairs) {
      std::vector< std::array<int, 2> > pairs2(intersections.pairs_found);
      cudaMemcpy(&pairs2[0], thrust::raw_pointer_cast(intersections.pairs.data()), sizeof(int2) * intersections.pairs_found, cudaMemcpyDeviceToHost);

      std::sort(pairs2.begin(), pairs2.end());

      for (int i = 0; i < pairs1.size(); i++) {
        EXPECT_EQ(pairs1[i], pairs2[i]);
      }
    }

  }

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
  BVH<dim> cpu_bvh(boxes_A);
  for (int j = 0; j < n; j++) {
    cpu_bvh.query(boxes_B[j], [&](int i){
      pairs1.push_back({i, j});
    });
  }

  std::sort(pairs1.begin(), pairs1.end());

  fm::AABB<dim> * d_boxes_B;
  cudaMalloc(&d_boxes_B, sizeof(fm::AABB<dim>) * n);
  cudaMemcpy(d_boxes_B, &boxes_B[0], sizeof(fm::AABB<dim>) * n, cudaMemcpyHostToDevice);

  std::cout << pairs1.size() << std::endl;

  int max_pairs = 1000000;
  GPU::intersection_list intersections(max_pairs);

  for (int k = 0; k < 5; k++) {

    GPU::BVH<dim> bvh(boxes_A, global_box);

    find_intersections(intersections, bvh, d_boxes_B, n);

    EXPECT_EQ(pairs1.size(), intersections.pairs_found);
    if (pairs1.size() == intersections.pairs_found && intersections.pairs_found <= max_pairs) {
      std::vector< std::array<int, 2> > pairs2(intersections.pairs_found);
      cudaMemcpy(&pairs2[0], thrust::raw_pointer_cast(intersections.pairs.data()), sizeof(int2) * intersections.pairs_found, cudaMemcpyDeviceToHost);

      std::sort(pairs2.begin(), pairs2.end());

      for (int i = 0; i < pairs1.size(); i++) {
        EXPECT_EQ(pairs1[i], pairs2[i]);
      }
    }

  }

  cudaFree(d_boxes_B);

}

TEST(UnitTest, BVHSelfIntersection2D) {
  for (int i = 0; i < 10; i++) {
    box_selfintersection_test<2>(1000, 0.02f);
  }
}

TEST(UnitTest, BVHSelfIntersection3D) {
  for (int i = 0; i < 10; i++) {
    box_selfintersection_test<3>(1000, 0.05f);
  }
}

TEST(UnitTest, BVHIntersection2D) {
  for (int i = 0; i < 10; i++) {
    box_intersection_test<2>(1000, 0.02f);
  }
}

TEST(UnitTest, BVHIntersection3D) {
  for (int i = 0; i < 10; i++) {
    box_intersection_test<3>(1000, 0.05f);
  }
}