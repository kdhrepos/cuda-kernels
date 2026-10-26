#include "../../common/cuda_check.cuh"
#include "../../common/gemm_report.cuh"
#include "../../common/gemm_tile.cuh"

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

#ifndef TM
#define TM 4
#endif

#ifndef TN
#define TN 4
#endif

void initMat(int row, int col, float* mat) {
    for (int r = 0; r < row; r++) {
        for (int c = 0; c < col; c++) {
            mat[r * col + c] = (float)(rand() % 10);
        }
    }
}

__global__ void coarsenedGemm(int M, int N, int K, float* A, float* B, float* C) {
    __shared__ float AsT[TILE][TILE * TM];
    __shared__ float Bs[TILE][TILE * TN];

    float acc[TM][TN] = {0.0f};
    float a[TM]       = {0.0f};
    float b[TN]       = {0.0f};

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    // This block owns TILE*TM rows and TILE*TN columns of C.
    const int blockRow = blockIdx.y * TILE * TM;
    const int blockCol = blockIdx.x * TILE * TN;

    for (int p = 0; p < gemmTiles(K, TILE); p++) {
        const int kTile = p * TILE;  // first k of this step

        // Load phase. The destination subscripts are swapped against the source
        // index -- that swap IS the transpose, and it is why the compute phase
        // below can read a column of A as a row of AsT.
        for (int j = 0; j < TM; j++)
            AsT[tx][ty + j * TILE] =
                gemmLoadOrZero(A, K, blockRow + ty + j * TILE, kTile + tx, M, K);

        for (int j = 0; j < TN; j++)
            Bs[ty][tx + j * TILE] =
                gemmLoadOrZero(B, N, kTile + ty, blockCol + tx + j * TILE, K, N);
        __syncthreads();

        // Compute phase: a different mapping of the same thread. It owns a
        // contiguous TM x TN block of C, and thanks to the transpose both a[]
        // and b[] now come from consecutive shared-memory addresses.
#pragma unroll
        for (int k = 0; k < TILE; k++) {
            for (int i = 0; i < TM; i++) a[i] = AsT[k][ty * TM + i];
            for (int i = 0; i < TN; i++) b[i] = Bs[k][tx * TN + i];

            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

    // Stores follow the compute mapping, not the load mapping.
    const int threadRow = blockRow + ty * TM;
    const int threadCol = blockCol + tx * TN;

    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++)
            gemmStoreIfInside(C, N, threadRow + i, threadCol + j, M, N, acc[i][j]);
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

    cudaEvent_t kernelStart, kernelEnd;
    CHECK_CUDA_ERROR(cudaEventCreate(&kernelStart));
    CHECK_CUDA_ERROR(cudaEventCreate(&kernelEnd));

    dim3 threadsPerBlock(TILE, TILE);
    dim3 blocksPerGrid((TILE * TN + N - 1) / (TILE * TN), (TILE * TM + M - 1) / (TILE * TM));

    GemmReport rep("thread_tile_4x4_transA", M, N, K, REPS);
    rep.setLaunch(blocksPerGrid, threadsPerBlock);
    int variant = rep.addVariant("4x4-transA");

    for (int i = 0; i <= REPS; i++) {
        CHECK_CUDA_ERROR(cudaEventRecord(kernelStart));
        coarsenedGemm<<<blocksPerGrid, threadsPerBlock>>>(M, N, K, d_a, d_b, d_c);
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

    free(h_a);
    free(h_b);
    free(h_c);

    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_c));

    return rep.passed() ? 0 : 1;
}