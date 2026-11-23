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
#define TM 8
#endif

#ifndef TN
#define TN 8
#endif

void initMat(int row, int col, float* mat) {
    for (int r = 0; r < row; r++) {
        for (int c = 0; c < col; c++) {
            mat[r * col + c] = (float)(rand() % 10);
        }
    }
}

__global__ void thread_tile_8x8_gemm(int M, int N, int K, float* A, float* B, float* C) {
    __shared__ float AsT[TN][TILE * TM];
    __shared__ float Bs[TM][TILE * TN];

    float acc[TM][TN] = {0.0f}; /* each thread has its own registers sized TMxTN */
    float a[TM]       = {0.0f};
    float b[TN]       = {0.0f};

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    // This block owns TILE*TM rows and TILE*TN columns of C. Deriving both
    // origins once is what keeps the row tile (TM) out of the column
    // arithmetic and vice versa.
    const int blockRow = blockIdx.y * TILE * TM;
    const int blockCol = blockIdx.x * TILE * TN;

    for (int p = 0; p < gemmTiles(K, TM); p++) {
        const int kTile = p * TM;  // first k of this step

        // Thread indices for tile As and Bs
        int aTileTy = tx / 2 + ty*TM;
        int aTileTx = tx % 2;
        int bTileTy = ty / 2;
        int bTileTx = (ty % 2)*TILE + tx;

        const float4 vA = 
            *reinterpret_cast<const float4*>(A + (blockRow + aTileTy)*K + kTile + aTileTx*4);
        AsT[aTileTx*4][aTileTy]     = vA.x;
        AsT[aTileTx*4 + 1][aTileTy] = vA.y;
        AsT[aTileTx*4 + 2][aTileTy] = vA.z;
        AsT[aTileTx*4 + 3][aTileTy] = vA.w;

        const float4 vB = 
            *reinterpret_cast<const float4*>(B + (kTile + bTileTy)*N + blockCol + bTileTx*4);
        *reinterpret_cast<float4*>(&Bs[bTileTy][bTileTx*4]) = vB;
        __syncthreads();

        // Compute phase
#pragma unroll
        for (int k = 0; k < TM; k++) {
            float4 a0 = *reinterpret_cast<float4*>(&AsT[k][ty*TM]);
            float4 a1 = *reinterpret_cast<float4*>(&AsT[k][ty*TM + 4]);
            float4 b0 = *reinterpret_cast<float4*>(&Bs[k][tx*TN]);
            float4 b1 = *reinterpret_cast<float4*>(&Bs[k][tx*TN + 4]);
            a[0] = a0.x, a[1] = a0.y, a[2] = a0.z, a[3] = a0.w;
            a[4] = a1.x, a[5] = a1.y, a[6] = a1.z, a[7] = a1.w;
            b[0] = b0.x, b[1] = b0.y, b[2] = b0.z, b[3] = b0.w;
            b[4] = b1.x, b[5] = b1.y, b[6] = b1.z, b[7] = b1.w;

#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < TN; j++) 
                    acc[i][j] += a[i] * b[j];
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

    GemmReport rep("thread_tile_8x8", M, N, K, REPS);
    rep.setLaunch(blocksPerGrid, threadsPerBlock);
    int variant = rep.addVariant("reg_tile_8x8");

    for (int i = 0; i <= REPS; i++) {
        CHECK_CUDA_ERROR(cudaEventRecord(kernelStart));
        thread_tile_8x8_gemm<<<blocksPerGrid, threadsPerBlock>>>(M, N, K, d_a, d_b, d_c);
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