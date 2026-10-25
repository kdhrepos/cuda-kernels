#pragma once

// cuBLAS status checking, split out of cuda_check.cuh so that only the
// benchmarks that actually race against cuBLAS have to link -lcublas.
//
//   #include "../../common/cublas_check.cuh"
//   CHECK_CUBLAS(cublasCreate(&handle));

#include <cublas_v2.h>

#include <iostream>
#include <cstdlib>

#define CHECK_CUBLAS(val) checkCublas((val), #val, __FILE__, __LINE__)
inline void checkCublas(cublasStatus_t status, char const* func, char const* file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cuBLAS Error at: " << file << ":" << line << std::endl;
        std::cerr << "status " << status << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
