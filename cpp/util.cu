#include "util.cuh"

#include <cuda.h>

#include "cub/cub.cuh"

#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/remove.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/device_ptr.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>

#define TO_THRUST(ptr) thrust::device_pointer_cast(ptr)

namespace util {

  void sort(
      void * buffer,
      size_t & size,
      uint32_t * keys,
      uint32_t * keys_sorted,
      float * values,
      float * values_sorted,
      int num_items,
      int bit_start,    
      int bit_end) {
    
    cub::DeviceRadixSort::SortPairs(
        buffer, size, 
        keys, keys_sorted, 
        values, values_sorted, 
        num_items, bit_start, bit_end);
    
  }

  void sort(
      void * buffer,
      size_t & size,
      uint64_t * keys,
      uint64_t * keys_sorted,
      float * values,
      float * values_sorted,
      int num_items,
      int bit_start,    
      int bit_end) {
    
    cub::DeviceRadixSort::SortPairs(
        buffer, size, 
        keys, keys_sorted, 
        values, values_sorted, 
        num_items, bit_start, bit_end);
    
  }

  void sort(
      void * buffer,
      size_t & size,
      uint32_t * keys,
      uint32_t * keys_sorted,
      double * values,
      double * values_sorted,
      int num_items,
      int bit_start,    
      int bit_end) {
    
    cub::DeviceRadixSort::SortPairs(
        buffer, size, 
        keys, keys_sorted, 
        values, values_sorted, 
        num_items, bit_start, bit_end);
    
  }

  void sort(
      void * buffer,
      size_t & size,
      uint64_t * keys,
      uint64_t * keys_sorted,
      double * values,
      double * values_sorted,
      int num_items,
      int bit_start,    
      int bit_end) {
    
    cub::DeviceRadixSort::SortPairs(
        buffer, size, 
        keys, keys_sorted, 
        values, values_sorted, 
        num_items, bit_start, bit_end);
    
  }

  void sort(
      void * buffer,
      size_t & size,
      uint64_t * keys,
      uint64_t * keys_sorted,
      int num_items,
      int bit_start,    
      int bit_end) {
    
    cub::DeviceRadixSort::SortKeys(
        buffer, size, 
        keys, keys_sorted, 
        num_items, bit_start, bit_end);
    
  }

  int reduce_by_key(
      uint32_t * keys_begin, 
      uint32_t * keys_end, 
      float * values_begin, 
      uint32_t * keys_reduced_begin, 
      float * values_reduced_begin) {
    
    // perform the reduction
    auto new_end = thrust::reduce_by_key(
        TO_THRUST(keys_begin),
        TO_THRUST(keys_end),
        TO_THRUST(values_begin),
        TO_THRUST(keys_reduced_begin),
        TO_THRUST(values_reduced_begin)
    );

    // and return the number of keys left
    return new_end.first - TO_THRUST(keys_reduced_begin);
    
  }

  int reduce_by_key(
      uint64_t * keys_begin, 
      uint64_t * keys_end, 
      float * values_begin, 
      uint64_t * keys_reduced_begin, 
      float * values_reduced_begin) {
    
    // perform the reduction
    auto new_end = thrust::reduce_by_key(
        TO_THRUST(keys_begin),
        TO_THRUST(keys_end),
        TO_THRUST(values_begin),
        TO_THRUST(keys_reduced_begin),
        TO_THRUST(values_reduced_begin)
    );

    // and return the number of keys left
    return new_end.first - TO_THRUST(keys_reduced_begin);
    
  }
  
  int reduce_by_key(
      uint32_t * keys_begin, 
      uint32_t * keys_end, 
      double * values_begin, 
      uint32_t * keys_reduced_begin, 
      double * values_reduced_begin) {
    
    // perform the reduction
    auto new_end = thrust::reduce_by_key(
        TO_THRUST(keys_begin),
        TO_THRUST(keys_end),
        TO_THRUST(values_begin),
        TO_THRUST(keys_reduced_begin),
        TO_THRUST(values_reduced_begin)
    );

    // and return the number of keys left
    return new_end.first - TO_THRUST(keys_reduced_begin);
    
  }
 
  int reduce_by_key(
      uint64_t * keys_begin, 
      uint64_t * keys_end, 
      double * values_begin, 
      uint64_t * keys_reduced_begin, 
      double * values_reduced_begin) {
    
    // perform the reduction
    auto new_end = thrust::reduce_by_key(
        TO_THRUST(keys_begin),
        TO_THRUST(keys_end),
        TO_THRUST(values_begin),
        TO_THRUST(keys_reduced_begin),
        TO_THRUST(values_reduced_begin)
    );

    // and return the number of keys left
    return new_end.first - TO_THRUST(keys_reduced_begin);
    
  } 

  void remove_flagged(
    	void * buffer,
    	size_t & size,
    	uint32_t * items,
    	uint32_t * flags,
    	uint32_t * items_compacted,
			int * remaining,
			int num_items) {

		cub::DeviceSelect::Flagged(
    	buffer,
    	size,
    	items,
    	flags,
    	items_compacted,
			remaining,
			num_items);

	}

  void remove_flagged(
    	void * buffer,
    	size_t & size,
    	uint64_t * items,
    	uint32_t * flags,
    	uint64_t * items_compacted,
			int * remaining,
			int num_items) {

		cub::DeviceSelect::Flagged(
    	buffer,
    	size,
    	items,
    	flags,
    	items_compacted,
			remaining,
			num_items);

	}

  void remove_flagged(
    	void * buffer,
    	size_t & size,
    	float * items,
    	uint32_t * flags,
    	float * items_compacted,
			int * remaining,
			int num_items) {

		cub::DeviceSelect::Flagged(
    	buffer,
    	size,
    	items,
    	flags,
    	items_compacted,
			remaining,
			num_items);

	}

  void remove_flagged(
    	void * buffer,
    	size_t & size,
    	double * items,
    	uint32_t * flags,
    	double * items_compacted,
			int * remaining,
			int num_items) {

		cub::DeviceSelect::Flagged(
    	buffer,
    	size,
    	items,
    	flags,
    	items_compacted,
			remaining,
			num_items);

	}

	void lower_bound(
			uint32_t * sorted_begin,
			uint32_t * sorted_end,
			uint32_t range_start,
			uint32_t range_end,
			uint32_t * indices) {

  	thrust::lower_bound(
  	    TO_THRUST(sorted_begin), 
  	    TO_THRUST(sorted_end),
  	    thrust::counting_iterator< uint32_t >(range_start),
  	    thrust::counting_iterator< uint32_t >(range_end),
  	    TO_THRUST(indices));

	}

	void lower_bound(
			int32_t * sorted_begin,
			int32_t * sorted_end,
			uint32_t range_start,
			uint32_t range_end,
			int32_t * indices) {

  	thrust::lower_bound(
  	    TO_THRUST(sorted_begin), 
  	    TO_THRUST(sorted_end),
  	    thrust::counting_iterator< uint32_t >(range_start),
  	    thrust::counting_iterator< uint32_t >(range_end),
  	    TO_THRUST(indices));

	}



}
