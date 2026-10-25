#pragma once

// Shared CUDA error checking for every benchmark in this repo.
//
//   #include "../common/cuda_check.cuh"      // from 01-vecadd/, reduction/, ...
//   #include "../../common/cuda_check.cuh"   // from gemm/<dir>/
//
//   CHECK_CUDA_ERROR(cudaMalloc(&d_a, bytes));  // any call returning cudaError_t
//   CHECK_LAST_CUDA_ERROR();                    // right after a kernel launch
//
// Both abort the process on the first failure: in a benchmark there is nothing
// useful to report once a CUDA call has failed.

#include <cuda_runtime.h>

#include <iostream>
#include <cstdlib>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
inline void check(cudaError_t err, char const* func, char const* file, int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
inline void checkLast(char const* file, int line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Reports a pending error and keeps going. 02-matadd and gray-to-scale do this
// on purpose -- they want to see the error without losing the rest of the run.
#define WARN_LAST_CUDA_ERROR() warnLast(__FILE__, __LINE__)
inline void warnLast(char const* file, int line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        // Deliberately no exit: the caller wants to continue.
    }
}

// This alias accepts the other word order so either spelling compiles.
#define CHECK_CUDA_LAST_ERROR() CHECK_LAST_CUDA_ERROR()
