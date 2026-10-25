#include "../../common/cuda_check.cuh"
#include "../../common/kernel_report.cuh"

#include <cuda_runtime.h>
#include <iostream>
#include <string>

// Measured iterations, after one discarded warm-up. Override at build time with
// -DREPS=n.
#ifndef REPS
#define REPS 10
#endif



#define BODY(j)                                         \
    for (int it = 0; it < iters; it++)                  \
        x = x * (0.999f - (j) * 1e-4f) + (j) * 1e-3f;

#define C1(j)  case (j): BODY(j) break;
#define C4(j)  C1(j) C1((j)+1) C1((j)+2) C1((j)+3)
#define C16(j) C4(j) C4((j)+4) C4((j)+8) C4((j)+12)

__global__ void threadDivergence(
    float *out,
    int iters,
    int n
) {
    int tid=blockDim.x*blockIdx.x+threadIdx.x;
    if (tid >= n) return;

    float x = (float)tid;
    if ((tid % 2) == 0) {
        BODY(0);
    } else {
        BODY(1);
    }
    out[tid]=x;
}

__global__ void warpDivergence(
    float *out,
    int iters,
    int n
) {
    int tid=blockDim.x*blockIdx.x+threadIdx.x;
    if (tid >= n) return;

    float x = (float)tid;
    if (((tid / 32) % 2) == 0) {
        BODY(0);
    } else {
        BODY(1);
    }
    out[tid]=x;
}

__global__ void kDivergence(
    float *a,
    int iters,
    int n,
    int k
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= n) return;

    float x = (float)tid;
    switch (tid % k) {
        C16(0)
        C16(16)
    }

    a[tid] = x;
}

__global__ void vecAddDiverge(
    float *a,
    float *b,
    float *c,
    int n
) {
    int tid=blockDim.x*blockIdx.x+threadIdx.x;

    if (tid < n) {
        if ((tid % 2) == 0) {
            c[tid] = a[tid]+b[tid];
        } else {
            c[tid] = a[tid]+b[tid] + 2;
        }
    }
}

int main(int argc, char** argv) {
    bool csv = bench::takeFlag(argc, argv, "--csv");
    int k, n, loopIters;
    
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <k> <n> <loopIters> [--csv]\n";
        return 0;
    } else {
        k=std::stoi(argv[1]);
        n=std::stoi(argv[2]);
        loopIters=std::stoi(argv[3]);
    }

    float* a=(float*)malloc(sizeof(float)*n);
    float* b=(float*)malloc(sizeof(float)*n);

    for (int i=0; i<n; i++) {
        a[i]=rand() % 5;
        b[i]=rand() % 5;
    }

    float* d_a;
    float* d_b;
    float* d_c;

    CHECK_CUDA_ERROR(cudaMalloc(&d_a, n*sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, n*sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_c, n*sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_a, a, n*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, b, n*sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t timerStart, timerEnd;
    CHECK_CUDA_ERROR(cudaEventCreate(&timerStart));
    CHECK_CUDA_ERROR(cudaEventCreate(&timerEnd));

    dim3 threadPerBlock(256);
    dim3 blocksPerGrid((n+255)/256);

    // warp-div is the reference: it branches, but uniformly within each warp, so
    // no lanes are masked off. The others show what divergence costs.
    KernelReport rep("divergence", REPS, 10);
    rep.addConfig("n", std::to_string(n) + " elements");
    rep.addConfig("k", std::to_string(k) + ", loop iters " + std::to_string(loopIters));
    rep.addConfig("launch", std::to_string(blocksPerGrid.x) + " blocks x "
                            + std::to_string(threadPerBlock.x) + " threads");
    rep.addIterations();
    int threadDiv = rep.addVariant("thread-div");
    int warpDiv   = rep.addVariant("warp-div");
    int kDiv      = rep.addVariant("k-div");
    int vecAdd    = rep.addVariant("vecadd");
    rep.setBaseline(warpDiv);

    for (int i=0; i<=REPS; i++) {
        float kernelTime=0.0;
        CHECK_CUDA_ERROR(cudaEventRecord(timerStart));
        threadDivergence<<<blocksPerGrid, threadPerBlock>>>(d_a, loopIters, n);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaEventRecord(timerEnd));
        CHECK_CUDA_ERROR(cudaEventSynchronize(timerEnd));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernelTime, timerStart, timerEnd));
        if (i!=0)
            rep.addSample(threadDiv, kernelTime);

        CHECK_CUDA_ERROR(cudaEventRecord(timerStart));
        warpDivergence<<<blocksPerGrid, threadPerBlock>>>(d_a, loopIters, n);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaEventRecord(timerEnd));
        CHECK_CUDA_ERROR(cudaEventSynchronize(timerEnd));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernelTime, timerStart, timerEnd));
        if (i!=0)
            rep.addSample(warpDiv, kernelTime);

        CHECK_CUDA_ERROR(cudaEventRecord(timerStart));
        kDivergence<<<blocksPerGrid, threadPerBlock>>>(d_a, loopIters, n, k);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaEventRecord(timerEnd));
        CHECK_CUDA_ERROR(cudaEventSynchronize(timerEnd));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernelTime, timerStart, timerEnd));
        if (i!=0)
            rep.addSample(kDiv, kernelTime);

        CHECK_CUDA_ERROR(cudaEventRecord(timerStart));
        vecAddDiverge<<<blocksPerGrid, threadPerBlock>>>(d_a, d_b, d_c, n);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaEventRecord(timerEnd));
        CHECK_CUDA_ERROR(cudaEventSynchronize(timerEnd));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernelTime, timerStart, timerEnd));
        if (i!=0)
            rep.addSample(vecAdd, kernelTime);
    }

    rep.print(csv);
}