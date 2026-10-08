/*
*	Field-dump gather kernels shared by the single- and multi-GPU CUDA engines
*	(defined in engine_cuda.cu). One CSR row per output value; see
*	Common/field_gather_backend.h.
*/
#ifndef ENGINE_CUDA_GATHER_CUH
#define ENGINE_CUDA_GATHER_CUH

#include "tools/constants.h"

#ifndef FDTD_FLOAT
#define FDTD_FLOAT float
#endif

#define DFT_FREQ_PER_LAUNCH 64
struct DFTWeights { float2 w[DFT_FREQ_PER_LAUNCH]; };

//! out[o] = sum over CSR entries [offsets[o],offsets[o+1]) of coeff*src[idx]
__global__ void fieldGatherKernel(const FDTD_FLOAT* __restrict__ src,
                                  const unsigned int* __restrict__ offsets,
                                  const unsigned int* __restrict__ idx,
                                  const float* __restrict__ coeff,
                                  float* __restrict__ out, int nOut);

//! the same gather, added to nf running-DFT sums (dft[k*nOut + o] += v*w[k])
__global__ void fieldGatherDFTKernel(const FDTD_FLOAT* __restrict__ src,
                                     const unsigned int* __restrict__ offsets,
                                     const unsigned int* __restrict__ idx,
                                     const float* __restrict__ coeff,
                                     float2* __restrict__ dft, int nOut,
                                     int nf, DFTWeights W);

#endif
