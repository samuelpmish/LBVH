#pragma once

#include "femto/tensor.h"
#include "misc/thrust_helper.h"

#include <variant>
#include <algorithm>

namespace Femto2D{

  class LineSegment{
   public:
    vec < 2, float > start, end;
  };

  class Triangle{
   public:
    vec < 2, float > vertices[3];
  };

  class Ball{
   public:
    vec < 2, float > center;
    float radius;
  };

  class AABB{
   public:
    float x_min, y_min;
    float x_max, y_max;

    __host__ __device__
    AABB(){};

    __host__ __device__
    AABB(double _x_min, double _y_min, double _x_max, double _y_max) {
      x_min = static_cast< float >(_x_min);
      y_min = static_cast< float >(_y_min);
      x_max = static_cast< float >(_x_max);
      y_max = static_cast< float >(_y_max);
    } 

    __host__ __device__
    AABB(float _x_min, float _y_min, float _x_max, float _y_max) {
      x_min = static_cast< float >(_x_min);
      y_min = static_cast< float >(_y_min);
      x_max = static_cast< float >(_x_max);
      y_max = static_cast< float >(_y_max);
    } 

    __host__ __device__
    explicit AABB(LineSegment L) {
      x_min = std::min(L.start[0], L.end[0]);
      y_min = std::min(L.start[1], L.end[1]);
      x_max = std::max(L.start[0], L.end[0]);
      y_max = std::max(L.start[1], L.end[1]);
    }

    __host__ __device__
    explicit AABB(Triangle t) {
      x_min = std::min(std::min(t.vertices[0][0], t.vertices[1][0]), t.vertices[2][0]);
      y_min = std::min(std::min(t.vertices[0][1], t.vertices[1][1]), t.vertices[2][1]);
      x_max = std::max(std::max(t.vertices[0][0], t.vertices[1][0]), t.vertices[2][0]);
      y_max = std::max(std::max(t.vertices[0][1], t.vertices[1][1]), t.vertices[2][1]);
    }

    __host__ __device__
    explicit AABB(Ball b) {
      x_min = b.center[0] - b.radius;
      y_min = b.center[1] - b.radius;
      x_max = b.center[0] + b.radius;
      y_max = b.center[1] + b.radius;
    }

    __host__ __device__
    vec < 2, float > center() const {
      return vec < 2, float >{
        (x_max + x_min) / static_cast< float >(2.0), 
        (y_max + y_min) / static_cast< float >(2.0)
      };
    }

    __host__ __device__
    vec < 2, float > widths() const {
      return vec < 2, float >{
        x_max - x_min, y_max - y_min
      };
    }

    __host__ __device__
    vec < 2, float > min() const {
      return vec < 2, float >{x_min, y_min};
    }

    __host__ __device__
    vec < 2, float > max() const {
      return vec < 2, float >{x_max, y_max}; 
    }

  };

  __host__ __device__
  inline AABB union_of(AABB a, AABB b) {
    return AABB(
      std::min(a.x_min, b.x_min),
      std::min(a.y_min, b.y_min),
      std::max(a.x_max, b.x_max),
      std::max(a.y_max, b.y_max)
    );
  }

  __host__ __device__
  inline AABB intersection_of(AABB a, AABB b) {
    return AABB(
      std::max(a.x_min, b.x_min),
      std::max(a.y_min, b.y_min),
      std::min(a.x_max, b.x_max),
      std::min(a.y_max, b.y_max)
    );
  }

  __host__ __device__
  inline bool intersecting(AABB a, AABB b) {
    return ((a.x_min <= b.x_max) & (a.x_max >= b.x_min) & 
            (a.y_min <= b.y_max) & (a.y_max >= b.y_min));
  }

  __host__ __device__
  inline bool intersecting(AABB a, vec2 b) {
    return ((a.x_min <= b[0]) & (b[0] <= a.x_max) & 
            (a.y_min <= b[1]) & (b[1] <= a.y_max));
  }

  using GeometryType = std::variant < LineSegment, Ball, Triangle, AABB >;

  template < MemorySpace M >
  class BVH {
   public:
    using code_t = uint64_t;

    AABB global;

    int32_t num_leaves;
    thrust::vector< int32_t, M > ids;
    thrust::vector< AABB, M > boxes;
    thrust::vector< int2, M > children;

    BVH(const thrust::vector< AABB, M > &);

    thrust::vector< int2, M > all_intersections_with(const thrust::vector < vec2 > & points);
  };

  using hBVH = BVH< MemorySpace::CPU >;

}