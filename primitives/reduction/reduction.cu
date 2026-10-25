#include "../../common/cuda_check.cuh"
#include "../../common/kernel_report.cuh"

#include <iostream>
#include <string>
#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>

// Measured iterations, after one discarded warm-up. Override at build time with
// -DREPS=n.
#ifndef REPS
#define REPS 10
#endif



__global__ void interleavedReduction(
    int N,
    float *in
) {
    int tx = threadIdx.x;
    int bx = blockDim.x*blockIdx.x;
    int tid = bx + tx;

    __shared__ float partial[256];
    partial[tx] = (tid < N) ? in[tid] : 0.0f;
    __syncthreads();

    for (int s=1; s<blockDim.x; s*=2) {
        if (tx % (s*2) == 0)
            partial[tx] += partial[tx + s];

        __syncthreads();
    }

    if (tx == 0)
        in[bx] = partial[0];
}

__global__ void sequentialReduction(
    int N,
    float *in
) {
    int tx = threadIdx.x;
    int bx = blockDim.x*blockIdx.x;
    int tid = bx + tx;

    __shared__ float partial[256];
    partial[tx] = (tid < N) ? in[tid] : 0.0f;
    __syncthreads();

    for (int s=blockDim.x/2; s>0; s/=2) {
        if (tx < s) 
            partial[tx] += partial[tx + s];
    
        __syncthreads();
    }

    if (tx == 0)
        in[bx] = partial[0];
}

__global__ void warpShuffledReduction(
    int N,
    float *in
) {
    int tx = threadIdx.x;
    int bx = blockDim.x*blockIdx.x;
    int tid = bx + tx;

    __shared__ float partial[8];
    int laneId = tx % 32;
    int warpId = tx / 32;
    float reg = (tid < N) ? in[tid] : 0.0f;

    for (int s=16; s>0; s/=2) {
        reg += __shfl_down_sync(0xFFFFFFFF, reg, s);
    }
    
    if (laneId == 0)
        partial[warpId] = reg;
    __syncthreads();
    
    if (warpId == 0) {
        reg = (tx < 8) ? partial[tx] : 0.0f;
        for (int s=4; s>0; s/=2) {
            reg += __shfl_down_sync(0xFFFFFFFF, reg, s);
        }
    }
    
    if (tx == 0)
        in[bx] = reg;
}

int checkBlocks(int N, int B, const float* h_in, const float* h_out) {
    int numBlocks = (N + B - 1) / B;
    int bad = 0;
    for (int b = 0; b < numBlocks; ++b) {
        float ref = 0.0f;
        int end = std::min(N, (b + 1) * B);
        for (int i = b * B; i < end; ++i) ref += h_in[i];
        float got = h_out[(size_t)b * B];
        if (got != ref) {
            if (bad < 5)
                std::cerr << "  block " << b << ": host " << ref << ", device " << got << "\n";
            ++bad;
        }
    }
    return bad;
}

int main(int argc, char** argv) {
    bool csv = bench::takeFlag(argc, argv, "--csv");
    int N;

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <N> [--csv]\n";
        return 1;
    } else { 
        N = std::stoi(argv[1]);
    }
    
    float *h_in = (float*)malloc(sizeof(float) * N);
    float *h_out = (float*)malloc(sizeof(float) * N);
    for (size_t i = 0; i < (size_t)N; ++i) h_in[i] = (float)(rand() % 10);

    float *d_in;
    CHECK_CUDA_ERROR(cudaMalloc(&d_in, sizeof(float) * N));

    cudaEvent_t kernelStart, kernelEnd;
    CHECK_CUDA_ERROR(cudaEventCreate(&kernelStart));
    CHECK_CUDA_ERROR(cudaEventCreate(&kernelEnd));

    float kernelTime=0.0;

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((256+N-1)/256);

    KernelReport rep("reduction", REPS, 11);
    rep.addConfig("N", std::to_string(N) + " elements ("
                       + bench::humanBytes(sizeof(float)*(double)N) + ")");
    rep.addConfig("launch", std::to_string(blocksPerGrid.x) + " blocks x "
                            + std::to_string(threadsPerBlock.x) + " threads");
    rep.addIterations();
    rep.addConfig("verified", "per-block sums vs CPU");
    int interleaved = rep.addVariant("interleaved");
    int sequential  = rep.addVariant("sequential");
    int shuffled    = rep.addVariant("warp-shuffle");
    // One pass over the input is the floor for a reduction, so that is the
    // traffic the bandwidth column is measured against.
    rep.setBytes(interleaved, sizeof(float)*(double)N);
    rep.setBytes(sequential,  sizeof(float)*(double)N);
    rep.setBytes(shuffled,    sizeof(float)*(double)N);
    rep.setBaseline(interleaved);

    for (int i=0; i<=REPS; i++) {
        CHECK_CUDA_ERROR(cudaMemcpy(d_in, h_in, sizeof(float) * N, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaEventRecord(kernelStart));
        interleavedReduction<<<blocksPerGrid, threadsPerBlock>>>(N, d_in);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaEventRecord(kernelEnd));
        CHECK_CUDA_ERROR(cudaEventSynchronize(kernelEnd));

        CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernelTime, kernelStart, kernelEnd));
        if (i==0) {
            CHECK_CUDA_ERROR(cudaMemcpy(h_out, d_in, sizeof(float) * N, cudaMemcpyDeviceToHost));
            rep.setCheck(interleaved, checkBlocks(N, 256, h_in, h_out));
        } else {
            rep.addSample(interleaved, kernelTime);
        }

        CHECK_CUDA_ERROR(cudaMemcpy(d_in, h_in, sizeof(float) * N, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaEventRecord(kernelStart));
        sequentialReduction<<<blocksPerGrid, threadsPerBlock>>>(N, d_in);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaEventRecord(kernelEnd));
        CHECK_CUDA_ERROR(cudaEventSynchronize(kernelEnd));

        CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernelTime, kernelStart, kernelEnd));
        if (i==0) {
            CHECK_CUDA_ERROR(cudaMemcpy(h_out, d_in, sizeof(float) * N, cudaMemcpyDeviceToHost));
            rep.setCheck(sequential, checkBlocks(N, 256, h_in, h_out));
        } else {
            rep.addSample(sequential, kernelTime);
        }

        CHECK_CUDA_ERROR(cudaMemcpy(d_in, h_in, sizeof(float) * N, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaEventRecord(kernelStart));
        warpShuffledReduction<<<blocksPerGrid, threadsPerBlock>>>(N, d_in);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaEventRecord(kernelEnd));
        CHECK_CUDA_ERROR(cudaEventSynchronize(kernelEnd));

        CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernelTime, kernelStart, kernelEnd));
        if (i==0) {
            CHECK_CUDA_ERROR(cudaMemcpy(h_out, d_in, sizeof(float) * N, cudaMemcpyDeviceToHost));
            rep.setCheck(shuffled, checkBlocks(N, 256, h_in, h_out));
        } else {
            rep.addSample(shuffled, kernelTime);
        }
    }

    rep.print(csv);

    return rep.passed() ? 0 : 1;
}