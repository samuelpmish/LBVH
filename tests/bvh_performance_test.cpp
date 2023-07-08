#include "AABB.hpp"
#include "BVH.hpp"

#define ANKERL_NANOBENCH_IMPLEMENT
#include "nanobench.h"

#include <array>
#include <random>
#include <iostream>

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
      float value = random_real();
      boxes[i].min[j] = value - 0.05f;
      boxes[i].max[j] = value + 0.05f;
    }
  }
  return boxes;
}

template < int dim >
int find_intersections_lbvh(const std::vector< AABB<dim> > & boxes) {
  int count = 0;
  BVH<dim> bvh(boxes);
  for (int i = 0; i < boxes.size(); i++) {
    bvh.query(boxes[i], [&](int j){ if (i < j) count++; });
  }
  return count;
}

template < int dim >
int find_intersections_brute_force(const std::vector< AABB<dim> > & boxes) {
  int count = 0;
  for (int i = 0; i < boxes.size(); i++) {
    for (int j = i+1; j < boxes.size(); j++) {
      if (intersecting(boxes[i], boxes[j])) {
        count++;
      }
    }
  }
  return count;
}

template < int dim >
void run_suite(ankerl::nanobench::Bench & bench) {
  for (int i = 0; i < 10; i++) {

    int n = 128 << i;
    auto boxes = random_AABBs<dim>(n);

    int count1 = -1;
    if (n < 4000) {
      bench.run("brute force " + std::to_string(n), [&] {
        count1 = find_intersections_brute_force(boxes); 
        ankerl::nanobench::doNotOptimizeAway(count1);
      });
    }

    int count2;
    bench.run("lbvh " + std::to_string(n), [&] {
      count2 = find_intersections_lbvh(boxes); 
      ankerl::nanobench::doNotOptimizeAway(count2);
    });

    if (count1 != -1 && count1 != count2) {
      std::cout << "error: disagreement in number of intersections" << std::endl;
    }
  }
}

int main() {
  ankerl::nanobench::Bench bench;
  run_suite<2>(bench);
  run_suite<3>(bench);
}

