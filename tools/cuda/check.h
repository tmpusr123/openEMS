
#pragma once

#include <stdio.h>
#include <assert.h>

#include "cuda_runtime_api.h"

namespace CudaHelper {

    inline void check_cuda(cudaError_t result)
    {
        if (result != cudaSuccess) {
            fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
            assert(result != cudaSuccess);
        }
    }

    inline void check_cuda()
    {
        check_cuda(cudaGetLastError()); // runtime API errors
    }
}