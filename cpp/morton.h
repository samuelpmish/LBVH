#pragma once

#ifndef __CUDACC__
#define __device__ 
#define __host__ 
#endif

#include <inttypes.h>

namespace morton {

  __host__ __device__
  inline uint32_t expand2(const uint32_t a) {

    uint32_t b = a; 

    b = (b | b << 16) & 0x0000FFFF; 
    b = (b | b <<  8) & 0x00FF00FF; 
    b = (b | b <<  4) & 0x0F0F0F0F; 
    b = (b | b <<  2) & 0x33333333; 
    b = (b | b <<  1) & 0x55555555;

    return b;

  }

  __host__ __device__
  inline uint64_t expand2(const uint64_t a) {

    uint64_t b = a; 

    b = (b | b << 32) & 0x00000000FFFFFFFF;
    b = (b | b << 16) & 0x0000FFFF0000FFFF;
    b = (b | b <<  8) & 0x00FF00FF00FF00FF;
    b = (b | b <<  4) & 0x0F0F0F0F0F0F0F0F;
    b = (b | b <<  2) & 0x3333333333333333;
    b = (b | b <<  1) & 0x5555555555555555;

    return b;
  }

  __host__ __device__
  inline uint32_t expand3(const uint32_t a) {

    uint32_t b = a; 

    b = (b * 0x00010001u) & 0xFF0000FFu;
    b = (b * 0x00000101u) & 0x0F00F00Fu;
    b = (b * 0x00000011u) & 0xC30C30C3u;
    b = (b * 0x00000005u) & 0x49249249u;

    return b;

  }

  __host__ __device__
  inline uint64_t expand3(const uint64_t a) {

    uint64_t b = a; 

    b = (b * 0x0000000100000001) & 0x001F00000000FFFF;
    b = (b * 0x0000000000010001) & 0x001F0000FF0000FF;
    b = (b * 0x0000000000000101) & 0x100F00F00F00F00F;
    b = (b * 0x0000000000000011) & 0x10C30C30C30C30C3;
    b = (b * 0x0000000000000005) & 0x1249249249249249;

    return b;

  }

  __host__ __device__
  inline uint32_t encode(const uint32_t x, const uint32_t y) {
    return (expand2(y) << 1) + expand2(x);
  }
  
  __host__ __device__
  inline uint64_t encode(const uint64_t x, const uint64_t y) {
    return (expand2(y) << 1) + expand2(x);
  }

  __host__ __device__
  inline uint32_t encode(const uint32_t x, const uint32_t y, const uint32_t z) {
    return (expand3(z) << 2) + (expand3(y) << 1) + expand3(x);
  }
  
  __host__ __device__
  inline uint64_t encode(const uint64_t x, const uint64_t y, const uint64_t z) {
    return (expand3(z) << 2) + (expand3(y) << 1) + expand3(x);
  }

}
