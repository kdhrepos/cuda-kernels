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



int checkCpu(
    int N,
    int *h_in,
    int *h_out
) {
    int mismatch_count = 0;
    for (int i=1; i<N; i++) {
        h_in[i] += h_in[i-1];

        if (h_in[i] != h_out[i])
           mismatch_count++;
    }

    return mismatch_count;
}

#define SCAN_BLOCK 256

size_t workspace_size(int n) {
    size_t total = 0;
    for (int m = n; m > 1; m = (m + SCAN_BLOCK - 1) / SCAN_BLOCK)
        total += (m + SCAN_BLOCK - 1) / SCAN_BLOCK;
    return total;
}

__global__ void block_scan(
    int n,
    const int *d_in,
    int *d_out,
    int *d_sums
) {
    __shared__ int smem[SCAN_BLOCK];

    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    smem[tx] = (tid < n) ? d_in[tid] : 0;
    __syncthreads();

    for (int d=1; d<SCAN_BLOCK; d*=2) {
        int r1, r2;
        if (tx >= d) {
            r1 = smem[tx];
            r2 = smem[tx-d];
        }
        __syncthreads();

        if (tx >= d) {
            smem[tx] = r1 + r2;
        }
        __syncthreads();
    }
    d_out[tid] = smem[tx];

    if (tx == SCAN_BLOCK-1)
        d_sums[bx] = smem[tx];
}

__global__ void add_block_sums(
    int n,
    int *d_out,
    const int *d_sums
) {
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    int bx = blockIdx.x;

    if (bx > 0 && tid < n) {
        d_out[tid] += d_sums[bx-1];
    }
}

void scan(
    const int* d_in,
    int* d_out,
    int n,
    int* d_ws
) {
    int blocks = (n + SCAN_BLOCK - 1) / SCAN_BLOCK;
    int* d_sums = d_ws;
    block_scan<<<blocks, SCAN_BLOCK>>>(n, d_in, d_out, d_ws);
    CHECK_LAST_CUDA_ERROR();
    if (blocks > 1) {
        scan(d_sums, d_sums, blocks, d_ws + blocks);
        add_block_sums<<<blocks, SCAN_BLOCK>>>(n, d_out, d_sums);
        CHECK_LAST_CUDA_ERROR();
    }
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

    int *h_in = (int *)malloc(sizeof(int) * N);
    int *h_out = (int *)malloc(sizeof(int) * N);
    for (int i=0; i<N; i++) h_in[i] = rand() % 256;

    int *d_in, *d_out, *d_ws;
    CHECK_CUDA_ERROR(cudaMalloc(&d_in, sizeof(int) * N));
    CHECK_CUDA_ERROR(cudaMalloc(&d_out, sizeof(int) * N));
    CHECK_CUDA_ERROR(cudaMalloc(&d_ws, sizeof(int) * workspace_size(N)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_in, h_in, sizeof(int) * N, cudaMemcpyHostToDevice));

    int blocks = (N + SCAN_BLOCK - 1) / SCAN_BLOCK;
    int* d_sums0 = d_ws;

    cudaEvent_t e0, e1, e2, e3;
    CHECK_CUDA_ERROR(cudaEventCreate(&e0));
    CHECK_CUDA_ERROR(cudaEventCreate(&e1));
    CHECK_CUDA_ERROR(cudaEventCreate(&e2));
    CHECK_CUDA_ERROR(cudaEventCreate(&e3));

    // The three kernels are stages of one scan, not rival implementations, so the
    // table sums them instead of ranking them against a baseline.
    KernelReport rep("scan", REPS, 12);
    rep.addConfig("N", std::to_string(N) + " elements ("
                       + bench::humanBytes(sizeof(int)*(double)N) + ")");
    rep.addConfig("launch", std::to_string(blocks) + " blocks x "
                            + std::to_string(SCAN_BLOCK) + " threads");
    rep.addIterations();
    rep.addConfig("verified", "full scan vs CPU");
    int blockScan    = rep.addVariant("block-scan");
    int recursive    = rep.addVariant("recursive");
    int addBlockSums = rep.addVariant("add-sums");
    double pass = 2.0 * sizeof(int) * (double)N;          // one read + one write
    rep.setBytes(blockScan,    pass);
    rep.setBytes(recursive,    2.0 * sizeof(int) * blocks);
    rep.setBytes(addBlockSums, pass);
    rep.setTotalRow("total");

    for (int i=0; i<=REPS; i++) {
        CHECK_CUDA_ERROR(cudaEventRecord(e0));
        block_scan<<<blocks, SCAN_BLOCK>>>(N, d_in, d_out, d_sums0);
        CHECK_CUDA_ERROR(cudaEventRecord(e1));
        scan(d_sums0, d_sums0, blocks, d_ws + blocks);
        CHECK_CUDA_ERROR(cudaEventRecord(e2));
        add_block_sums<<<blocks, SCAN_BLOCK>>>(N, d_out, d_sums0);
        CHECK_CUDA_ERROR(cudaEventRecord(e3));
        CHECK_CUDA_ERROR(cudaEventSynchronize(e3));

        float kernelTime[3] = {0.0f, 0.0f, 0.0f};
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernelTime[0], e0, e1));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernelTime[1], e1, e2));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernelTime[2], e2, e3));
        if (i==0) {
            CHECK_CUDA_ERROR(cudaMemcpy(h_out, d_out, sizeof(int) * N, cudaMemcpyDeviceToHost));
            rep.setTotalCheck(checkCpu(N, h_in, h_out));
        } else {
            rep.addSample(blockScan,    kernelTime[0]);
            rep.addSample(recursive,    kernelTime[1]);
            rep.addSample(addBlockSums, kernelTime[2]);
        }
    }

    rep.print(csv);

    CHECK_CUDA_ERROR(cudaEventDestroy(e0));
    CHECK_CUDA_ERROR(cudaEventDestroy(e1));
    CHECK_CUDA_ERROR(cudaEventDestroy(e2));
    CHECK_CUDA_ERROR(cudaEventDestroy(e3));

    free(h_in);
    free(h_out);

    CHECK_CUDA_ERROR(cudaFree(d_in));
    CHECK_CUDA_ERROR(cudaFree(d_out));

    return rep.passed() ? 0 : 1;
}