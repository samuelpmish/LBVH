#pragma once

#include <fstream>
#include <iostream>

#include "femto/defines.h"

#include "thrust/remove.h"
#include "thrust/device_ptr.h"
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"

template < typename T >
using hvector = thrust::host_vector< T >;

template < typename T >
using dvector = thrust::device_vector< T >;

template < typename T >
T * raw_ptr(thrust::host_vector < T > & hvec) {
  return thrust::raw_pointer_cast(hvec.data());
}

template < typename T >
const T * raw_ptr(const thrust::host_vector < T > & hvec) {
  return thrust::raw_pointer_cast(hvec.data());
}

template < typename T >
T * raw_ptr(thrust::device_vector < T > & dvec) {
  return thrust::raw_pointer_cast(dvec.data());
}

template < typename T >
const T * raw_ptr(const thrust::device_vector < T > & dvec) {
  return thrust::raw_pointer_cast(dvec.data());
}

template < typename T, typename U >
const U * raw_ptr_cast(const hvector < T > & hvec) {
  return reinterpret_cast< const U * >(thrust::raw_pointer_cast(hvec.data()));
}

template < typename U, typename T >
U * raw_ptr_cast(hvector < T > & hvec) {
  return reinterpret_cast< U * >(thrust::raw_pointer_cast(hvec.data()));
}

template < typename T >
struct true_enough {
  __host__ __device__
  bool operator()(const T & b) const { return b; }
};

namespace thrust {

  template < typename T, MemorySpace M = MemorySpace::CPU >
  using vector = typename std::conditional< M == MemorySpace::CPU, 
        hvector< T >, 
        dvector< T > >::type;

  template < typename value_type, typename flag_type >
  void remove_flagged(hvector< value_type > & v, hvector< flag_type > & flags) {
    auto new_end = remove_if(v.begin(), v.end(), flags.begin(), true_enough< flag_type >());
    v.resize(new_end - v.begin());
  } 

  template < typename value_type, typename flag_type >
  void remove_flagged(dvector< value_type > & v, dvector< flag_type > & flags) {
    auto new_end = remove_if(v.begin(), v.end(), flags.begin(), true_enough< flag_type >());
    v.resize(new_end - v.begin());
  } 

}

template < typename T >
hvector < T > read_binary(std::string filename) {

  hvector < T > buffer;

  std::ifstream infile(filename, std::ios::binary);

  if (infile) {
    infile.seekg(0,std::ios::end);
    std::streampos filesize = infile.tellg();
    infile.seekg(0,std::ios::beg);

    buffer = hvector < T >(filesize / sizeof(T));
    infile.read((char*)&buffer[0], filesize);
  } else {
    std::cout << "file not found: " << filename << std::endl;
  }

  infile.close();

  return buffer;
  
}

template < typename T >
void write_binary(const hvector < T > & buffer, std::string filename) {

  std::ofstream outfile(filename, std::ios::binary);

  if (outfile) {
    outfile.write((char*)&buffer[0], sizeof(T) * buffer.size());
  } else {
    std::cout << "file not found: " << filename << std::endl;
  }

  outfile.close();
  
}

#ifndef __CUDACC__
struct int2{
  int32_t x, y;
};
#endif

inline bool operator<(const int2 & a, const int2 & b) { 
  return (a.x < b.x) || (a.x == b.x && a.y < b.y);
}