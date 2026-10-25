#include "../../common/cuda_check.cuh"
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

#ifndef TILE
#define TILE 16
#endif



__global__ void tiledGemm(
    int M,
    int N,
    int K,
    float* A,
    float* B,
    float* C
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    float sum = 0.0f;
    for (int p = 0; p < (K+TILE-1)/TILE; p++) {
        As[ty][tx] = ((p*TILE+tx < K) && (row < M)) ? A[row*K + p*TILE+tx] : 0.0f;
        Bs[ty][tx] = ((ty + p*TILE < K) && (col < N)) ? B[(ty+p*TILE)*N + col] : 0.0f;
        __syncthreads();

        for (int i = 0; i < TILE; i++) {
            sum += As[ty][i]*Bs[i][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) 
        C[row*N+col] = sum;
}

void initMat(
    int row, 
    int col, 
    float* mat) 
{
    for (int r=0; r<row; r++) {
        for (int c=0; c<col; c++) {
            mat[r*col + c] = (float)(rand() % 10);
        }
    }
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

    size_t size_a=sizeof(float)*M*K;
    size_t size_b=sizeof(float)*K*N;
    size_t size_c=sizeof(float)*M*N;

    float *h_a=(float *)malloc(size_a);
    float *h_b=(float *)malloc(size_b);
    float *h_c=(float *)malloc(size_c);

    initMat(M, K, h_a);
    initMat(K, N, h_b);
    
    float *d_a, *d_b, *d_c;
    CHECK_CUDA_ERROR(cudaMalloc(&d_a, size_a));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, size_b));
    CHECK_CUDA_ERROR(cudaMalloc(&d_c, size_c));
    
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice));

    cudaEvent_t kernelStart, kernelEnd;
    CHECK_CUDA_ERROR(cudaEventCreate(&kernelStart));
    CHECK_CUDA_ERROR(cudaEventCreate(&kernelEnd));


    dim3 threadsPerBlock(TILE, TILE);
    dim3 blocksPerGrid((TILE+N-1)/TILE, (TILE+M-1)/TILE);

    GemmReport rep("tiled_gemm", M, N, K, REPS);
    rep.setLaunch(blocksPerGrid, threadsPerBlock);
    int variant = rep.addVariant("tiled");

    for (int i=0; i<=REPS; i++) {
        CHECK_CUDA_ERROR(cudaEventRecord(kernelStart));
        tiledGemm<<<blocksPerGrid, threadsPerBlock>>>(M, N, K, d_a, d_b, d_c);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaEventRecord(kernelEnd));
        CHECK_CUDA_ERROR(cudaEventSynchronize(kernelEnd));

        float kernelTime=0.0;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernelTime, kernelStart, kernelEnd));
        // i == 0 is the warm-up: verified, then thrown away.
        if (i==0) {
            CHECK_CUDA_ERROR(cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost));
            rep.setCheck(variant, gemmCheckCpu(M, N, K, h_a, h_b, h_c));
        } else {
            rep.addSample(variant, kernelTime);
        }
    }

    rep.print(csv);

    CHECK_CUDA_ERROR(cudaEventDestroy(kernelStart));
    CHECK_CUDA_ERROR(cudaEventDestroy(kernelEnd));

    free(h_a);
    free(h_b);
    free(h_c);

    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_c));

    return rep.passed() ? 0 : 1;
}