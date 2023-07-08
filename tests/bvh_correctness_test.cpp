#include <gtest/gtest.h>

#include "AABB.hpp"
#include "BVH.hpp"

#include <random>

float random_real() {
  static std::default_random_engine generator;
  static std::uniform_real_distribution<float> distribution(-1.0,1.0);
  return distribution(generator);
};

template < int dim >
std::vector< AABB<dim> > random_AABBs(int n) {
  std::vector< AABB<dim> > boxes(n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < dim; j++) {
      float values[2] = {random_real(), random_real()};
      boxes[i].min[j] = std::min(values[0], values[1]);
      boxes[i].max[j] = std::max(values[0], values[1]);
    }
  }
  return boxes;
}

template < int dim >
void run_test(int n) {

  auto boxes = random_AABBs<dim>(n);

  BVH<dim> bvh(boxes);
  std::vector< std::array<int, 2> > pairs1;
  for (int i = 0; i < n; i++) {
    bvh.query(boxes[i], [&](int j){ if (i < j) pairs1.push_back({i, j}); });
  }

  std::vector< std::array<int, 2> > pairs2;
  for (int i = 0; i < n; i++) {
    for (int j = i+1; j < n; j++) {
      if (intersecting(boxes[i], boxes[j])) {
        pairs2.push_back({i, j});
      }
    }
  }

  EXPECT_EQ(pairs1.size(), pairs2.size());

  std::sort(pairs1.begin(), pairs1.end());
  std::sort(pairs2.begin(), pairs2.end());

  for (int i = 0; i < pairs1.size(); i++) {
    EXPECT_EQ(pairs1[i], pairs2[i]);
  }

}

TEST(UnitTest, BVHIntersection2D) {
  run_test<2>(1000);
}

TEST(UnitTest, BVHIntersection3D) {
  run_test<3>(1000);
}
