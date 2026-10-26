#include "../../common/cuda_check.cuh"
#include "../../common/kernel_report.cuh"

#include <iostream>
#include <string>
#include <cmath>

#include <cuda_runtime.h>

// Measured iterations, after one discarded warm-up. Override at build time with
// -DREPS=n.
#ifndef REPS
#define REPS 10
#endif

#define TB_SIZE 1556
#define SM_NUM 82

#define NUM_BIN 256

__global__ void global_histogram(int N, unsigned char* vec, unsigned int* bin) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < N) {
        unsigned char v = vec[tid];
        atomicAdd(&bin[v], 1);
    }
}

__global__ void shared_histogram(int N, unsigned char* vec, unsigned int* bin) {
    int tx         = threadIdx.x;
    int block_size = blockDim.x;
    int block_idx  = blockIdx.x;
    int grid_size  = gridDim.x;

    int elems_per_thread =
        (N + grid_size * block_size - 1) / (grid_size * block_size);  // elements per thread

    __shared__ unsigned int bin_s[NUM_BIN];
    bin_s[tx] = 0;
    __syncthreads();

    for (int i = 0; i < elems_per_thread; i++) {
        int idx = block_size * grid_size * i + block_size * block_idx + tx;

        if (idx < N) {
            unsigned char v = vec[idx];
            atomicAdd(&bin_s[v], 1);
        }
    }
    __syncthreads();

    atomicAdd(&bin[tx], bin_s[tx]);
}

int checkCpu(int N, unsigned char* h_vec, unsigned int* h_bin) {
    int bin[NUM_BIN] = {
        0,
    };
    for (int i = 0; i < N; i++) bin[h_vec[i]]++;

    int mismatch_count = 0;
    for (int i = 0; i < NUM_BIN; i++)
        if (h_bin[i] != bin[i]) mismatch_count++;

    return mismatch_count;
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

    unsigned char* h_vec = (unsigned char*)malloc(sizeof(unsigned char) * N);
    unsigned int* h_bin  = (unsigned int*)malloc(sizeof(unsigned int) * NUM_BIN);
    for (int i = 0; i < N; i++) h_vec[i] = rand() % NUM_BIN;

    unsigned char* d_vec;
    unsigned int* d_bin;
    CHECK_CUDA_ERROR(cudaMalloc(&d_vec, sizeof(unsigned char) * N));
    CHECK_CUDA_ERROR(cudaMalloc(&d_bin, sizeof(unsigned int) * NUM_BIN));
    CHECK_CUDA_ERROR(cudaMemcpy(d_vec, h_vec, sizeof(unsigned char) * N, cudaMemcpyHostToDevice));

    cudaEvent_t kernelStart, kernelEnd;
    CHECK_CUDA_ERROR(cudaEventCreate(&kernelStart));
    CHECK_CUDA_ERROR(cudaEventCreate(&kernelEnd));

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((256 + N - 1) / 256);
    dim3 blocksPerGridV2(492);

    KernelReport rep("histogram", REPS, 8);
    rep.addConfig("N", std::to_string(N) + " bytes (" + bench::humanBytes((double)N) + "), " +
                           std::to_string(NUM_BIN) + " bins");
    // The two kernels deliberately use different grids: one block per chunk of
    // input vs a fixed grid that loops, so each block keeps its own shared bins.
    rep.addConfig("launch", "global " + std::to_string(blocksPerGrid.x) + " blocks, shared " +
                                std::to_string(blocksPerGridV2.x) + " blocks, " +
                                std::to_string(threadsPerBlock.x) + " threads");
    rep.addIterations();
    rep.addConfig("verified", "all bins vs CPU");
    int globalBins = rep.addVariant("global");
    int sharedBins = rep.addVariant("shared");
    rep.setBytes(globalBins, (double)N);  // one pass over the input
    rep.setBytes(sharedBins, (double)N);
    rep.setBaseline(globalBins);

    for (int i = 0; i <= REPS; i++) {
        CHECK_CUDA_ERROR(cudaMemset(d_bin, 0, sizeof(unsigned int) * NUM_BIN));
        CHECK_CUDA_ERROR(cudaEventRecord(kernelStart));
        global_histogram<<<blocksPerGrid, threadsPerBlock>>>(N, d_vec, d_bin);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaEventRecord(kernelEnd));
        CHECK_CUDA_ERROR(cudaEventSynchronize(kernelEnd));

        float kernelTime = 0.0;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernelTime, kernelStart, kernelEnd));
        if (i == 0) {
            CHECK_CUDA_ERROR(
                cudaMemcpy(h_bin, d_bin, sizeof(unsigned int) * NUM_BIN, cudaMemcpyDeviceToHost));
            rep.setCheck(globalBins, checkCpu(N, h_vec, h_bin));
        } else {
            rep.addSample(globalBins, kernelTime);
        }

        CHECK_CUDA_ERROR(cudaMemset(d_bin, 0, sizeof(unsigned int) * NUM_BIN));
        CHECK_CUDA_ERROR(cudaEventRecord(kernelStart));
        shared_histogram<<<blocksPerGridV2, threadsPerBlock>>>(N, d_vec, d_bin);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaEventRecord(kernelEnd));
        CHECK_CUDA_ERROR(cudaEventSynchronize(kernelEnd));

        kernelTime = 0.0;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernelTime, kernelStart, kernelEnd));
        if (i == 0) {
            CHECK_CUDA_ERROR(
                cudaMemcpy(h_bin, d_bin, sizeof(unsigned int) * NUM_BIN, cudaMemcpyDeviceToHost));
            rep.setCheck(sharedBins, checkCpu(N, h_vec, h_bin));
        } else {
            rep.addSample(sharedBins, kernelTime);
        }
    }

    rep.print(csv);

    CHECK_CUDA_ERROR(cudaEventDestroy(kernelStart));
    CHECK_CUDA_ERROR(cudaEventDestroy(kernelEnd));

    free(h_vec);
    free(h_bin);

    CHECK_CUDA_ERROR(cudaFree(d_vec));
    CHECK_CUDA_ERROR(cudaFree(d_bin));

    return rep.passed() ? 0 : 1;
}