#include "../../common/cuda_check.cuh"
#include "../../common/kernel_report.cuh"

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <string>
#include <cmath>

// Measured iterations, after one discarded warm-up. Override at build time with
// -DREPS=n.
#ifndef REPS
#define REPS 10
#endif

#define ROW 9999
#define COL 6001

__global__ void matAdd(float* d_a, float* d_b, float* d_c) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx < COL && ty < ROW) {
        d_c[ty * COL + tx] = d_a[ty * COL + tx] + d_b[ty * COL + tx];
    }
}

int main(int argc, char** argv) {
    bool csv       = bench::takeFlag(argc, argv, "--csv");
    int bx         = 16;
    int by         = 16;
    size_t matSize = sizeof(float) * ROW * COL;

    if (argc == 3) {
        bx = std::stoi(argv[1]);
        by = std::stoi(argv[2]);
    }

    float* h_a = (float*)malloc(matSize);
    float* h_b = (float*)malloc(matSize);
    float* h_c = (float*)malloc(matSize);

    // init
    for (int r = 0; r < ROW; r++) {
        for (int c = 0; c < COL; c++) {
            h_a[r * COL + c] = (float)rand() / RAND_MAX * 1e6f - 5e5f;
            h_b[r * COL + c] = (float)rand() / RAND_MAX * 1e6f - 5e5f;
        }
    }

    float *d_a, *d_b, *d_c;
    CHECK_CUDA_ERROR(cudaMalloc(&d_a, matSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, matSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_c, matSize));

    // host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, matSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, matSize, cudaMemcpyHostToDevice));

    cudaEvent_t kernelStart, kernelEnd;
    CHECK_CUDA_ERROR(cudaEventCreate(&kernelStart));
    CHECK_CUDA_ERROR(cudaEventCreate(&kernelEnd));

    // launch config
    dim3 threadsPerBlock(bx, by);
    dim3 blocksPerGrid((COL + bx - 1) / bx, (ROW + by - 1) / by);

    KernelReport rep("matadd", REPS, 7);
    rep.addConfig("matrix", std::to_string(ROW) + " x " + std::to_string(COL) + " floats (" +
                                bench::humanBytes((double)matSize) + "/matrix)");
    rep.addConfig("launch", std::to_string(blocksPerGrid.x) + "x" +
                                std::to_string(blocksPerGrid.y) + " blocks, " + std::to_string(bx) +
                                "x" + std::to_string(by) + " threads");
    rep.addIterations();
    rep.addConfig("verified", "every element vs CPU");
    int matadd = rep.addVariant("matadd");
    rep.setBytes(matadd, 3.0 * matSize);  // a and b read, c written

    for (int i = 0; i <= REPS; i++) {
        // launch kernel
        CHECK_CUDA_ERROR(cudaEventRecord(kernelStart));
        matAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);
        WARN_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaEventRecord(kernelEnd));
        CHECK_CUDA_ERROR(cudaEventSynchronize(kernelEnd));

        float totalKernelTime = 0.0;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&totalKernelTime, kernelStart, kernelEnd));

        if (i == 0) {
            // device to host
            CHECK_CUDA_ERROR(cudaMemcpy(h_c, d_c, matSize, cudaMemcpyDeviceToHost));

            long long mismatches = 0;
            for (int r = 0; r < ROW; r++) {
                for (int c = 0; c < COL; c++) {
                    float sum = h_a[r * COL + c] + h_b[r * COL + c];

                    if (fabs(h_c[r * COL + c] - sum) > 1e-7) mismatches++;
                }
            }
            rep.setCheck(matadd, mismatches);
        } else {
            rep.addSample(matadd, totalKernelTime);
        }
    }

    CHECK_CUDA_ERROR(cudaEventDestroy(kernelStart));
    CHECK_CUDA_ERROR(cudaEventDestroy(kernelEnd));

    rep.print(csv);

    free(h_a);
    free(h_b);
    free(h_c);

    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_c));

    return rep.passed() ? 0 : 1;
}
