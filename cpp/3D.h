#pragma once

#include "femto/tensor.h"
#include "misc/thrust_helper.h"

#include <variant>
#include <algorithm>

namespace Femto3D{

  class LineSegment{
   public:
    vec < 3, float > start, end;
  };

  class Triangle{
   public:
    vec < 3, float > vertices[3];
  };

  class Ball{
   public:
    vec < 3, float > center;
    float radius;
  };

  class AABB{
   public:
    float x_min, y_min, z_min;
    float x_max, y_max, z_max;

    __host__ __device__
    AABB(){};

    __host__ __device__
    AABB(float _x_min, 
         float _y_min, 
         float _z_min, 
         float _x_max, 
         float _y_max, 
         float _z_max) {
      x_min = static_cast< float >(_x_min); 
      y_min = static_cast< float >(_y_min); 
      z_min = static_cast< float >(_z_min);
      x_max = static_cast< float >(_x_max); 
      y_max = static_cast< float >(_y_max); 
      z_max = static_cast< float >(_z_max);
    } 

    __host__ __device__
    explicit AABB(LineSegment L) {
      x_min = std::min(L.start[0], L.end[0]);
      y_min = std::min(L.start[1], L.end[1]);
      z_min = std::min(L.start[2], L.end[2]);
      x_max = std::max(L.start[0], L.end[0]);
      y_max = std::max(L.start[1], L.end[1]);
      z_max = std::max(L.start[2], L.end[2]);
    }

    __host__ __device__
    explicit AABB(Triangle t) {
      x_min = std::min(std::min(t.vertices[0][0], t.vertices[1][0]), t.vertices[2][0]);
      y_min = std::min(std::min(t.vertices[0][1], t.vertices[1][1]), t.vertices[2][1]);
      z_min = std::min(std::min(t.vertices[0][2], t.vertices[1][2]), t.vertices[2][2]);
      x_max = std::max(std::max(t.vertices[0][0], t.vertices[1][0]), t.vertices[2][0]);
      y_max = std::max(std::max(t.vertices[0][1], t.vertices[1][1]), t.vertices[2][1]);
      z_max = std::max(std::max(t.vertices[0][2], t.vertices[1][2]), t.vertices[2][2]);
    }

    __host__ __device__
    explicit AABB(Ball b) {
      x_min = b.center[0] - b.radius;
      y_min = b.center[1] - b.radius;
      z_min = b.center[2] - b.radius;
      x_max = b.center[0] + b.radius;
      y_max = b.center[1] + b.radius;
      z_max = b.center[2] + b.radius;
    }

    __host__ __device__
    vec < 3, float > center() const {
      return vec < 3, float >{
        (x_max + x_min) / float(2.0), 
        (y_max + y_min) / float(2.0), 
        (z_max + z_min) / float(2.0)
      };
    }

    __host__ __device__
    vec < 3, float > widths() const {
      return vec < 3, float >{
        x_max - x_min, y_max - y_min, z_max - z_min
      };
    }

    __host__ __device__
    vec < 3, float > min() const {
      return vec < 3, float >{x_min, y_min, z_min};
    }

    __host__ __device__
    vec < 3, float > max() const {
      return vec < 3, float >{x_max, y_max, z_max}; 
    }

  };

  __host__ __device__
  inline AABB union_of(AABB a, AABB b) {
    return AABB(
      std::min(a.x_min, b.x_min),
      std::min(a.y_min, b.y_min),
      std::min(a.z_min, b.z_min),
      std::max(a.x_max, b.x_max),
      std::max(a.y_max, b.y_max),
      std::max(a.z_max, b.z_max)
    );
  }

  __host__ __device__
  inline AABB intersection_of(AABB a, AABB b) {
    return AABB(
      std::max(a.x_min, b.x_min),
      std::max(a.y_min, b.y_min),
      std::max(a.z_min, b.z_min),
      std::min(a.x_max, b.x_max),
      std::min(a.y_max, b.y_max),
      std::min(a.z_max, b.z_max)
    );
  }

  __host__ __device__
  inline bool intersecting(AABB a, AABB b) {
    return ((a.x_min <= b.x_max) & (a.x_max >= b.x_min) & 
            (a.y_min <= b.y_max) & (a.y_max >= b.y_min) & 
            (a.z_min <= b.z_max) & (a.z_max >= b.z_min));
  }

  __host__ __device__
  inline bool intersecting(AABB a, vec3 b) {
    return ((a.x_min <= b[0]) & (b[0] <= a.x_max) & 
            (a.y_min <= b[1]) & (b[1] <= a.y_max) & 
            (a.z_min <= b[2]) & (b[2] <= a.z_max));
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

    thrust::vector< int2, M > all_intersections_with(const thrust::vector < vec3 > & points);
  };

  using hBVH = BVH< MemorySpace::CPU >;

}