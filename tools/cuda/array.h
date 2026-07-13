#pragma once

#include "cuda_runtime_api.h"

#include "check.h"

namespace CudaHelper {
    template <typename T> class Array;

    template <typename T>
    class Array 
    {
    public:
        Array(size_t n)
        {
            extPtr = NULL;
            checkCuda(cudaHostAlloc(&hPtr, n * sizeof(T), cudaHostAllocDefault));
            checkCuda(cudaMalloc(&dPtr, n * sizeof(T)));
            nSize = n;
        }

        Array(size_t n, T* host)  // array attach to an existing host pointer
        {
            extPtr = host;
            hPtr = NULL;
            checkCuda(cudaMalloc(&dPtr, n * sizeof(T)));
            nSize = n;
        }

        ~Array() 
        {
            if (dPtr) check_cuda(cudaFree(dPtr));
            if (hPtr) check_cuda(cudaFreeHost(hPtr));  // hPtr came from cudaHostAlloc
        }

        size_t size()   const {return nSize; }
        size_t bytes()   const {return nSize * sizeof(T); }
        T* device_data() {return dPtr;}
        T* host_data() {return hPtr;}
        T at(size_t n) {return hPtr[n];}

        void inline clear() 
        {
            if (hPtr) memset(hPtr, 0, bytes());
            if (extPtr) memset(extPtr, 0, bytes());
            checkCuda(cudaMemset(dPtr, 0, bytes()));
        }

        void load_to_host() {
            T *dest = (hPtr) ? hPtr : extPtr;
            checkCuda(cudaMemcpy(dest, dPtr, bytes(), cudaMemcpyDeviceToHost));
        }

        void load_to_host(T *dest) {
            checkCuda(cudaMemcpy(dest, dPtr, bytes(), cudaMemcpyDeviceToHost));
        }

        void load_to_host_async() {
            checkCuda(cudaMemcpyAsync(hPtr, dPtr, bytes(), cudaMemcpyDeviceToHost));
        }

        void load_to_device() {
            T *src = (hPtr) ? hPtr : extPtr;
            checkCuda(cudaMemcpy(dPtr, src, bytes(), cudaMemcpyHostToDevice));
        }

        void load_to_device(T *src) {
            checkCuda(cudaMemcpy(dPtr, src, bytes(), cudaMemcpyHostToDevice));
        }

        void load_to_device_async() {
            checkCuda(cudaMemcpyAsync(dPtr, hPtr, bytes(), cudaMemcpyHostToDevice));
        }

    private:
        size_t nSize;
        T *hPtr;
        T *dPtr;
        T *extPtr;
    };
}