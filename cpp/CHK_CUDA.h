#pragma once

//------------------------------------------------------------------------------
#define CHK_CUDA( code )                                                       \
{                                                                              \
  cudaError_t err__ = code;                                                    \
  if( err__ != cudaSuccess )                                                   \
  {                                                                            \
    std::cerr << "Error in file " << __FILE__ << " on line " << __LINE__ << ":"\
              << cudaGetErrorString( err__ ) << std::endl;                     \
    exit(1);                                                                   \
  }                                                                            \
}
//------------------------------------------------------------------------------
