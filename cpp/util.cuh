#pragma once

#include <inttypes.h>

namespace util {

  void sort(
      void * buffer,
      size_t & size,
      uint32_t * keys,
      uint32_t * keys_sorted,
      float * values,
      float * values_sorted,
      int num_items,
      int bit_start = 0,
      int bit_end = 32);

  void sort(
      void * buffer,
      size_t & size,
      uint64_t * keys,
      uint64_t * keys_sorted,
      float * values,
      float * values_sorted,
      int num_items,
      int bit_start = 0,
      int bit_end = 64);

  void sort(
      void * buffer,
      size_t & size,
      uint32_t * keys,
      uint32_t * keys_sorted,
      double * values,
      double * values_sorted,
      int num_items,
      int bit_start = 0,
      int bit_end = 32);

  void sort(
      void * buffer,
      size_t & size,
      uint64_t * keys,
      uint64_t * keys_sorted,
      double * values,
      double * values_sorted,
      int num_items,
      int bit_start = 0,
      int bit_end = 64);

  void sort(
      void * buffer,
      size_t & size,
      uint64_t * keys,
      uint64_t * keys_sorted,
      int num_items,
      int bit_start = 0,
      int bit_end = 64);

  int reduce_by_key(
      uint32_t * keys_begin, 
      uint32_t * keys_end, 
      float * values_begin, 
      uint32_t * keys_reduced_begin, 
      float * values_reduced_begin);

  int reduce_by_key(
      uint64_t * keys_begin, 
      uint64_t * keys_end, 
      float * values_begin, 
      uint64_t * keys_reduced_begin, 
      float * values_reduced_begin);

  int reduce_by_key(
      uint32_t * keys_begin, 
      uint32_t * keys_end, 
      double * values_begin, 
      uint32_t * keys_reduced_begin, 
      double * values_reduced_begin);

  int reduce_by_key(
      uint64_t * keys_begin, 
      uint64_t * keys_end, 
      double * values_begin, 
      uint64_t * keys_reduced_begin, 
      double * values_reduced_begin);

  void remove_flagged(
      void * buffer,
      size_t & size,
      uint32_t * items,
      uint32_t * flags,
      uint32_t * items_compacted,
      int * remaining,
      int num_items);

  void remove_flagged(
      void * buffer,
      size_t & size,
      uint64_t * items,
      uint32_t * flags,
      uint64_t * items_compacted,
      int * remaining,
      int num_items);

  void remove_flagged(
      void * buffer,
      size_t & size,
      float * items,
      uint32_t * flags,
      float * items_compacted,
      int * remaining,
      int num_items);

  void remove_flagged(
      void * buffer,
      size_t & size,
      double * items,
      uint32_t * flags,
      double * items_compacted,
      int * remaining,
      int num_items);

  void lower_bound(
      uint32_t * sorted_begin,
      uint32_t * sorted_end,
      uint32_t range_start,
      uint32_t range_end,
      uint32_t * indices);

  void lower_bound(
      int32_t * sorted_begin,
      int32_t * sorted_end,
      uint32_t range_start,
      uint32_t range_end,
      int32_t * indices);

  void sum();

}
