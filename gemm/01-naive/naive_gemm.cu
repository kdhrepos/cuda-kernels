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



__global__ void naiveGemm(
    int m,
    int n,
    int k,
    float* a,
    float* b,
    float* c
) {
    int tx=blockDim.x*blockIdx.x+threadIdx.x;
    int ty=blockDim.y*blockIdx.y+threadIdx.y;

    if (ty < m && tx < n) {
        float sum=0.0;
        for (int i=0; i<k; i++) {
            sum+=a[ty*k+i]*b[i*n+tx];
        }
        c[ty*n+tx]=sum;
    }
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
    int bx, by;
    int m, n, k;

    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <bx> <by> <M> <N> <K> [--csv]\n";
        return 1;
    } else {
        bx = std::stoi(argv[1]);
        by = std::stoi(argv[2]);
        m = std::stoi(argv[3]);
        n = std::stoi(argv[4]);
        k = std::stoi(argv[5]);
    }

    size_t size_a=sizeof(float)*m*k;
    size_t size_b=sizeof(float)*k*n;
    size_t size_c=sizeof(float)*m*n;

    float *h_a=(float *)malloc(size_a);
    float *h_b=(float *)malloc(size_b);
    float *h_c=(float *)malloc(size_c);

    initMat(m, k, h_a);
    initMat(k, n, h_b);

    float *d_a, *d_b, *d_c;
    CHECK_CUDA_ERROR(cudaMalloc(&d_a, size_a));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, size_b));
    CHECK_CUDA_ERROR(cudaMalloc(&d_c, size_c));

    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice));

    cudaEvent_t kernelStart, kernelEnd;
    CHECK_CUDA_ERROR(cudaEventCreate(&kernelStart));
    CHECK_CUDA_ERROR(cudaEventCreate(&kernelEnd));


    dim3 threadsPerBlock(bx, by);
    dim3 blocksPerGrid((bx+n-1)/bx, (by+m-1)/by);

    GemmReport rep("naive_gemm", m, n, k, REPS);
    rep.setLaunch(blocksPerGrid, threadsPerBlock);
    int variant = rep.addVariant("naive");

    for (int i=0; i<=REPS; i++) {
        CHECK_CUDA_ERROR(cudaEventRecord(kernelStart));
        naiveGemm<<<blocksPerGrid, threadsPerBlock>>>(m, n, k, d_a, d_b, d_c);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaEventRecord(kernelEnd));
        CHECK_CUDA_ERROR(cudaEventSynchronize(kernelEnd));

        float kernelTime=0.0;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernelTime, kernelStart, kernelEnd));
        // i == 0 is the warm-up: verified, then thrown away.
        if (i==0) {
            CHECK_CUDA_ERROR(cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost));
            rep.setCheck(variant, gemmCheckCpu(m, n, k, h_a, h_b, h_c));
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