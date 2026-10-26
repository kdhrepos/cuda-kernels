#include "../../common/cuda_check.cuh"
#include "../../common/kernel_report.cuh"

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
#define TILE 32
#endif

double gbps(int M, int N, float ms) {
    double bytes = 2.0 * M * N * sizeof(float);
    return bytes / (ms * 1e-3) / 1e9;  // = bytes / ms / 1e6
}

__global__ void copyOnlyKernel(int M, int N, float* in, float* out) {
    int tiy = blockDim.y * blockIdx.y + threadIdx.y;
    int tix = blockDim.x * blockIdx.x + threadIdx.x;

    if (tiy < M && tix < N) out[tiy * N + tix] = in[tiy * N + tix];
}

__global__ void naiveReadCoalescedMatrixTranspose(int M, int N, float* in, float* out) {
    int tiy = blockDim.y * blockIdx.y + threadIdx.y;
    int tix = blockDim.x * blockIdx.x + threadIdx.x;

    if (tiy < M && tix < N) out[tix * M + tiy] = in[tiy * N + tix];
}

__global__ void naiveWriteCoalescedMatrixTranspose(int M, int N, float* in, float* out) {
    int tiy = blockDim.y * blockIdx.y + threadIdx.y;
    int tix = blockDim.x * blockIdx.x + threadIdx.x;

    if (tiy < N && tix < M) out[tiy * M + tix] = in[tix * N + tiy];
}

__global__ void tiledMatrixTranspose(int M, int N, float* in, float* out) {
    __shared__ float ins[TILE][TILE];

    int tiy   = threadIdx.y;
    int tix   = threadIdx.x;
    int inRow = blockDim.y * blockIdx.y + tiy;
    int inCol = blockDim.x * blockIdx.x + tix;

    if (inRow < M && inCol < N) ins[tiy][tix] = in[inRow * N + inCol];
    __syncthreads();

    int outRow = blockDim.x * blockIdx.x + tiy;
    int outCol = blockDim.y * blockIdx.y + tix;

    if (outRow < N && outCol < M) out[outRow * M + outCol] = ins[tix][tiy];
}

__global__ void tiledMatrixTransposePadded(int M, int N, float* in, float* out) {
    __shared__ float ins[TILE][TILE + 1];

    int tiy   = threadIdx.y;
    int tix   = threadIdx.x;
    int inRow = blockDim.y * blockIdx.y + tiy;
    int inCol = blockDim.x * blockIdx.x + tix;

    if (inRow < M && inCol < N) ins[tiy][tix] = in[inRow * N + inCol];
    __syncthreads();

    int outRow = blockDim.x * blockIdx.x + tiy;
    int outCol = blockDim.y * blockIdx.y + tix;

    if (outRow < N && outCol < M) out[outRow * M + outCol] = ins[tix][tiy];
}

void initMat(int row, int col, float* mat) {
    int i = 0;
    for (int r = 0; r < row; r++)
        for (int c = 0; c < col; c++) mat[r * col + c] = i++;
}

int checkCpu(int M, int N, float* in, float* out) {
    int mismatchCount = 0;

    for (int r = 0; r < M; r++)
        for (int c = 0; c < N; c++)
            if (in[r * N + c] != out[c * M + r]) mismatchCount++;
    // std::cerr << "Mismatch at: " << r << ", " << c << std::endl;

    return mismatchCount;
}

int main(int argc, char** argv) {
    bool csv = bench::takeFlag(argc, argv, "--csv");
    int M, N;

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <M> <N> [--csv]\n";
        return 1;
    } else {
        M = std::stoi(argv[1]);
        N = std::stoi(argv[2]);
    }

    size_t size = sizeof(float) * M * N;

    float* h_in  = (float*)malloc(size);
    float* h_out = (float*)malloc(size);

    initMat(M, N, h_in);

    float *d_in, *d_out;
    CHECK_CUDA_ERROR(cudaMalloc(&d_in, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_out, size));

    CHECK_CUDA_ERROR(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    cudaEvent_t kernelStart, kernelEnd;
    CHECK_CUDA_ERROR(cudaEventCreate(&kernelStart));
    CHECK_CUDA_ERROR(cudaEventCreate(&kernelEnd));

    float kernelTime = 0.0;
    dim3 threadsPerBlock(TILE, TILE);

    // copy-only is not a transpose; it is the bandwidth ceiling the transposes
    // are measured against, so it is the baseline and it is not verified.
    KernelReport rep("transpose", REPS, 10);
    rep.addConfig("matrix", std::to_string(M) + " x " + std::to_string(N) + " floats (" +
                                bench::humanBytes((double)size) + ")");
    rep.addConfig("launch",
                  "tiles of " + std::to_string(TILE) + " x " + std::to_string(TILE) + " threads");
    rep.addIterations();
    rep.addConfig("verified", "every element vs CPU");
    int copyOnly   = rep.addVariant("copy-only");
    int readCoal   = rep.addVariant("read-coal");
    int tiled      = rep.addVariant("tiled");
    int tiledPad   = rep.addVariant("tiled-pad");
    int writeCoal  = rep.addVariant("write-coal");
    double traffic = 2.0 * M * N * sizeof(float);  // one read + one write
    rep.setBytes(copyOnly, traffic);
    rep.setBytes(readCoal, traffic);
    rep.setBytes(tiled, traffic);
    rep.setBytes(tiledPad, traffic);
    rep.setBytes(writeCoal, traffic);
    rep.setBaseline(copyOnly);

    for (int i = 0; i <= REPS; i++) {
        dim3 blocksPerGrid((TILE + N - 1) / TILE, (TILE + M - 1) / TILE);

        CHECK_CUDA_ERROR(cudaMemset(d_out, 0xFF, size));
        CHECK_CUDA_ERROR(cudaEventRecord(kernelStart));
        copyOnlyKernel<<<blocksPerGrid, threadsPerBlock>>>(M, N, d_in, d_out);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaEventRecord(kernelEnd));
        CHECK_CUDA_ERROR(cudaEventSynchronize(kernelEnd));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernelTime, kernelStart, kernelEnd));
        if (i != 0) rep.addSample(copyOnly, kernelTime);

        CHECK_CUDA_ERROR(cudaMemset(d_out, 0xFF, size));
        CHECK_CUDA_ERROR(cudaEventRecord(kernelStart));
        naiveReadCoalescedMatrixTranspose<<<blocksPerGrid, threadsPerBlock>>>(M, N, d_in, d_out);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaEventRecord(kernelEnd));
        CHECK_CUDA_ERROR(cudaEventSynchronize(kernelEnd));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernelTime, kernelStart, kernelEnd));
        if (i == 0) {
            CHECK_CUDA_ERROR(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));
            rep.setCheck(readCoal, checkCpu(M, N, h_in, h_out));
        } else {
            rep.addSample(readCoal, kernelTime);
        }

        CHECK_CUDA_ERROR(cudaMemset(d_out, 0xFF, size));
        CHECK_CUDA_ERROR(cudaEventRecord(kernelStart));
        tiledMatrixTranspose<<<blocksPerGrid, threadsPerBlock>>>(M, N, d_in, d_out);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaEventRecord(kernelEnd));
        CHECK_CUDA_ERROR(cudaEventSynchronize(kernelEnd));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernelTime, kernelStart, kernelEnd));
        if (i == 0) {
            CHECK_CUDA_ERROR(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));
            rep.setCheck(tiled, checkCpu(M, N, h_in, h_out));
        } else {
            rep.addSample(tiled, kernelTime);
        }

        CHECK_CUDA_ERROR(cudaMemset(d_out, 0xFF, size));
        CHECK_CUDA_ERROR(cudaEventRecord(kernelStart));
        tiledMatrixTransposePadded<<<blocksPerGrid, threadsPerBlock>>>(M, N, d_in, d_out);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaEventRecord(kernelEnd));
        CHECK_CUDA_ERROR(cudaEventSynchronize(kernelEnd));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernelTime, kernelStart, kernelEnd));
        if (i == 0) {
            CHECK_CUDA_ERROR(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));
            rep.setCheck(tiledPad, checkCpu(M, N, h_in, h_out));
        } else {
            rep.addSample(tiledPad, kernelTime);
        }

        blocksPerGrid = dim3((TILE + M - 1) / TILE, (TILE + N - 1) / TILE);

        CHECK_CUDA_ERROR(cudaMemset(d_out, 0xFF, size));
        CHECK_CUDA_ERROR(cudaEventRecord(kernelStart));
        naiveWriteCoalescedMatrixTranspose<<<blocksPerGrid, threadsPerBlock>>>(M, N, d_in, d_out);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaEventRecord(kernelEnd));
        CHECK_CUDA_ERROR(cudaEventSynchronize(kernelEnd));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernelTime, kernelStart, kernelEnd));
        if (i == 0) {
            CHECK_CUDA_ERROR(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));
            rep.setCheck(writeCoal, checkCpu(M, N, h_in, h_out));
        } else {
            rep.addSample(writeCoal, kernelTime);
        }
    }

    rep.print(csv);

    CHECK_CUDA_ERROR(cudaEventDestroy(kernelStart));
    CHECK_CUDA_ERROR(cudaEventDestroy(kernelEnd));

    free(h_in);
    free(h_out);

    CHECK_CUDA_ERROR(cudaFree(d_in));
    CHECK_CUDA_ERROR(cudaFree(d_out));

    return rep.passed() ? 0 : 1;
}