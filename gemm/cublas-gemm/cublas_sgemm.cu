#include "../../common/cuda_check.cuh"
#include "../../common/cublas_check.cuh"
#include "../../common/gemm_report.cuh"

#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <cmath>

// Measured iterations, after one discarded warm-up. Override at build time with
// -DREPS=n.
#ifndef REPS
#define REPS 10
#endif

void initMat(int row, int col, float* mat) {
    for (int r = 0; r < row; r++) {
        for (int c = 0; c < col; c++) {
            mat[r * col + c] = (float)(rand() % 10);
        }
    }
}

// cuBLAS reads column-major, while every matrix in this repo is row-major. Rather
// than transposing anything, ask cuBLAS for C^T = B^T * A^T: a row-major M x N
// matrix is exactly an N x M column-major one, so swapping the operands and
// passing the row lengths as leading dimensions computes the row-major product
// with no data movement.
void launchSgemm(cublasHandle_t handle, int M, int N, int K, const float* d_a, const float* d_b,
                 float* d_c) {
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_b, N, d_a, K,
                             &beta, d_c, N));
}

int main(int argc, char** argv) {
    bool csv = bench::takeFlag(argc, argv, "--csv");

    int M, N, K;

    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <M> <N> <K> [--csv]\n";
        return 1;
    } else {
        M = std::stoi(argv[1]);
        N = std::stoi(argv[2]);
        K = std::stoi(argv[3]);
    }

    size_t size_a = sizeof(float) * M * K;
    size_t size_b = sizeof(float) * K * N;
    size_t size_c = sizeof(float) * M * N;

    float* h_a = (float*)malloc(size_a);
    float* h_b = (float*)malloc(size_b);
    float* h_c = (float*)malloc(size_c);

    initMat(M, K, h_a);
    initMat(K, N, h_b);

    float *d_a, *d_b, *d_c;
    CHECK_CUDA_ERROR(cudaMalloc(&d_a, size_a));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, size_b));
    CHECK_CUDA_ERROR(cudaMalloc(&d_c, size_c));

    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    cudaEvent_t kernelStart, kernelEnd;
    CHECK_CUDA_ERROR(cudaEventCreate(&kernelStart));
    CHECK_CUDA_ERROR(cudaEventCreate(&kernelEnd));

    GemmReport rep("cublas_sgemm", M, N, K, REPS);
    // cuBLAS picks its own grid and block, so there is no launch config to report.
    int variant = rep.addVariant("cublas");

    for (int i = 0; i <= REPS; i++) {
        CHECK_CUDA_ERROR(cudaEventRecord(kernelStart));
        launchSgemm(handle, M, N, K, d_a, d_b, d_c);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaEventRecord(kernelEnd));
        CHECK_CUDA_ERROR(cudaEventSynchronize(kernelEnd));

        float kernelTime = 0.0;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernelTime, kernelStart, kernelEnd));
        // i == 0 is the warm-up: verified, then thrown away.
        if (i == 0) {
            CHECK_CUDA_ERROR(cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost));
            rep.setCheck(variant, gemmCheckCpu(M, N, K, h_a, h_b, h_c));
        } else {
            rep.addSample(variant, kernelTime);
        }
    }

    rep.print(csv);

    CHECK_CUDA_ERROR(cudaEventDestroy(kernelStart));
    CHECK_CUDA_ERROR(cudaEventDestroy(kernelEnd));
    CHECK_CUBLAS(cublasDestroy(handle));

    free(h_a);
    free(h_b);
    free(h_c);

    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_c));

    return rep.passed() ? 0 : 1;
}
