#include <gtest/gtest.h>

#include "AABB.hpp"

TEST(UnitTest, AABB2D) {

  AABB<2> a{{0.0f, 0.0f}, {1.0f, 1.0f}};
  AABB<2> b{{0.5f, 0.5f}, {1.5f, 1.5f}};
  AABB<2> c{{1.0f, 0.0f}, {2.0f, 1.0f}};
  AABB<2> d{{1.1f, 0.0f}, {2.1f, 1.0f}};
  AABB<2> e{{1.0f, 1.0f}, {2.0f, 2.0f}};

  EXPECT_TRUE(intersecting(a, b));
  EXPECT_TRUE(intersecting(b, c));
  EXPECT_FALSE(intersecting(a, d)); 

  // pathological case where intersection is 1-dimensional
  EXPECT_TRUE(intersecting(a, c)); 
  EXPECT_TRUE(intersection_of(a, c).min[0] == 1.0f);
  EXPECT_TRUE(intersection_of(a, c).max[0] == 1.0f);

  // pathological case where intersection is 0-dimensional
  EXPECT_TRUE(intersecting(a, e)); 
  EXPECT_TRUE(intersection_of(a, e).min[0] == 1.0f);
  EXPECT_TRUE(intersection_of(a, e).max[0] == 1.0f);
  EXPECT_TRUE(intersection_of(a, e).min[1] == 1.0f);
  EXPECT_TRUE(intersection_of(a, e).max[1] == 1.0f);

  EXPECT_TRUE(bounding_box(a, c).min[0] == 0.0f);
  EXPECT_TRUE(bounding_box(a, c).max[0] == 2.0f);

}

TEST(UnitTest, AABBIntersection3D) {

  AABB<3> a{{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}};
  AABB<3> b{{0.5f, 0.5f, 0.5f}, {1.5f, 1.5f, 1.5f}};
  AABB<3> c{{1.0f, 1.0f, 0.0f}, {2.0f, 1.0f, 1.0f}};
  AABB<3> d{{1.0f, 1.0f, 0.0f}, {2.0f, 2.0f, 2.0f}};
  AABB<3> e{{1.0f, 1.0f, 1.0f}, {2.0f, 2.0f, 2.0f}};
  AABB<3> f{{1.1f, 1.1f, 1.1f}, {2.1f, 2.1f, 2.1f}};

  EXPECT_TRUE(intersecting(a, b));

  // pathological case where intersection is 2-dimensional
  EXPECT_TRUE(intersecting(a, c)); 
  EXPECT_TRUE(intersection_of(a, c).min[0] == 1.0f);
  EXPECT_TRUE(intersection_of(a, c).max[0] == 1.0f);

  // pathological case where intersection is 1-dimensional
  EXPECT_TRUE(intersecting(a, d)); 
  EXPECT_TRUE(intersection_of(a, d).min[0] == 1.0f);
  EXPECT_TRUE(intersection_of(a, d).max[0] == 1.0f);
  EXPECT_TRUE(intersection_of(a, d).min[1] == 1.0f);
  EXPECT_TRUE(intersection_of(a, d).max[1] == 1.0f);

  // pathological case where intersection is 0-dimensional
  EXPECT_TRUE(intersecting(a, e)); 
  EXPECT_TRUE(intersection_of(a, e).min[0] == 1.0f);
  EXPECT_TRUE(intersection_of(a, e).max[0] == 1.0f);
  EXPECT_TRUE(intersection_of(a, e).min[1] == 1.0f);
  EXPECT_TRUE(intersection_of(a, e).max[1] == 1.0f);
  EXPECT_TRUE(intersection_of(a, e).min[2] == 1.0f);
  EXPECT_TRUE(intersection_of(a, e).max[2] == 1.0f);

  EXPECT_FALSE(intersecting(a, f)); 

  EXPECT_TRUE(bounding_box(a, c).min[0] == 0.0f);
  EXPECT_TRUE(bounding_box(a, c).max[0] == 2.0f);

}
