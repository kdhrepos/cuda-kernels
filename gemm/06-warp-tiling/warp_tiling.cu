#include "../../common/cuda_check.cuh"
#include "../../common/gemm_report.cuh"
#include "../../common/gemm_tile.cuh"

#include <cuda_runtime.h>
#include <iostream>

// Measured iterations, after one discarded warm-up. Override at build time with
// -DREPS=n.
#ifndef REPS
#define REPS 20
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

#ifndef BM
#define BM ((TILE)*(TM))
#endif

#ifndef BN
#define BN ((TILE)*(TN))
#endif

#ifndef BK
#define BK 8
#endif

#ifndef N_BUF
#define N_BUF 2
#endif

constexpr int TY = BM / TM;
constexpr int TX = BN / TN;
constexpr int N_THREADS = TX * TY;

static_assert(TM % 4 == 0 && TN % 4 == 0, "TM, TN must be multiples of 4 (float4)");
static_assert(N_THREADS <= 1024, "too many threads per block");
static_assert((BM * BK / 4) % N_THREADS == 0, "A vectors must split evenly over threads");

__global__ void sgemm(int M, int N, int K, float* A, float* B, float* C) {
    // double buffer for read and write
    __shared__ float AsT[N_BUF][BK][BM];
    __shared__ float Bs[N_BUF][BK][BN];

    float acc[TM][TN] = {0.0f}; /* each thread has its own registers sized TMxTN */
    float a[TM]       = {0.0f};
    float b[TN]       = {0.0f};

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int tid = ty*blockDim.x + tx;

    // This block owns TILE*TM rows and TILE*TN columns of C. Deriving both
    // origins once is what keeps the row tile (TM) out of the column
    // arithmetic and vice versa.
    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    const int warpRowSize = TM * 4; /* number of element that a warp will process for one row */
    const int warpColSize = TN * 8; /* number of element that a warp will process for one column */
    const int warpRowNum = BM / warpRowSize; /* number of warps in the row of a thread block */
    const int warpColNum = BN / warpColSize; /* number of warps in the column of a thread block */

    // Used for warp tiling
    const int warpId = tid / 32;
    const int laneId = tid % 32;
    const int warpRow = warpId / warpColNum;
    const int warpCol = warpId % warpColNum;
    const int laneRow = laneId / 8; /* lane dim is fixed as 4x8 */
    const int laneCol = laneId % 8;

    // number of iterations to load data from global memory
    constexpr int aLoadNum = (BM * BK) / (4*N_THREADS); /* 4 is the number of element to load at a time */ 
    constexpr int bLoadNum = (BK * BN) / (4*N_THREADS); /* 4 is the number of element to load at a time */ 

    // buffer index for double buffering
    int buf = 0;

    // vector arrays for load data from global memory
    float4 vA[aLoadNum], vB[bLoadNum];

    // prologue (first compute phase)
#pragma unroll
    for (int i = 0; i < aLoadNum; i++) {
        // divide BK by 4 since each thread load four data at a time 
        const int stride = i * TY * TX;
        const int vid = tid + stride; /* vector id to load */
        const int aLoadRow = vid / (BK/4);
        const int aLoadCol = vid % (BK/4);
        vA[i] = *reinterpret_cast<float4*>(A + (blockRow + aLoadRow)*K + aLoadCol*4);

        AsT[buf][aLoadCol*4][aLoadRow]      = vA[i].x;
        AsT[buf][aLoadCol*4 + 1][aLoadRow]  = vA[i].y;
        AsT[buf][aLoadCol*4 + 2][aLoadRow]  = vA[i].z;
        AsT[buf][aLoadCol*4 + 3][aLoadRow]  = vA[i].w;
    }

#pragma unroll
for (int i = 0; i < bLoadNum; i++) {
        const int stride = i * TY * TX;
        const int vid = tid + stride; /* vector id to load */
        const int bLoadRow = vid / (BN/4);
        const int bLoadCol = vid % (BN/4);
        vB[i] = *reinterpret_cast<float4*>(B + (bLoadRow)*N + blockCol + bLoadCol*4);

        *reinterpret_cast<float4*>(&Bs[buf][bLoadRow][bLoadCol*4]) = vB[i];
    }
    __syncthreads();

    /* steady state */
    for (int p = 1; p < gemmTiles(K, BK); p++) { /* no load at last iteration */
        const int kTile = p * BK;  /* first k of this step */

        // load first, do not store the data at this time for latency hiding
#pragma unroll
        for (int i = 0; i < aLoadNum; i++) {
            const int stride = i * TY * TX;
            const int vid = tid + stride; /* vector id to load */
            const int aLoadRow = vid / (BK/4);
            const int aLoadCol = vid % (BK/4);
            vA[i] = *reinterpret_cast<float4*>(A + (blockRow + aLoadRow)*K + kTile + aLoadCol*4);
        }

#pragma unroll
        for (int i = 0; i < bLoadNum; i++) {
            const int stride = i * TY * TX;
            const int vid = tid + stride; /* vector id to load */
            const int bLoadRow = vid / (BN/4);
            const int bLoadCol = vid % (BN/4);
            vB[i] = *reinterpret_cast<float4*>(B + (kTile + bLoadRow)*N + blockCol + bLoadCol*4);
        }

        // compute phase
#pragma unroll
        for (int k = 0; k < BK; k++) {
            // Load A from AsT to register a.
            // Each thread need to iterate for TM/4 times.
            // The reason why divide TM by 4 is each thread load four elements at a time.
            // If TM is 8, each thread loads one vector, two times. 
#pragma unroll
            for (int vecIdx = 0; vecIdx < TM/4; vecIdx++) {
                // 16 = row size of a warp * number of element to load at a time (4 * 4)
                // 4 is the number of element to load at a time
                float4 vecA = *reinterpret_cast<float4*>(&AsT[buf][k][warpRow*warpRowSize + vecIdx*16 + laneRow*4]);

                a[vecIdx*4]      = vecA.x;
                a[vecIdx*4 + 1]  = vecA.y;
                a[vecIdx*4 + 2]  = vecA.z;
                a[vecIdx*4 + 3]  = vecA.w;
            }
            
            // load B from Bs to register b
#pragma unroll
            for (int vecIdx = 0; vecIdx < TN/4; vecIdx++) {
                // 32 = col size of a warp * number of element to load at a time (8 * 4)
                // 4 is the number of element to load at a time
                float4 vecB = *reinterpret_cast<float4*>(&Bs[buf][k][warpCol*warpColSize + vecIdx*32 + laneCol*4]);
                
                b[vecIdx*4]      = vecB.x;
                b[vecIdx*4 + 1]  = vecB.y;
                b[vecIdx*4 + 2]  = vecB.z;
                b[vecIdx*4 + 3]  = vecB.w;
            }

#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < TN; j++) 
                    acc[i][j] += a[i] * b[j];
        }

        // store after the computation
        buf = (buf+1) % N_BUF;
#pragma unroll
        for (int i = 0; i < aLoadNum; i++) {
            const int stride = i * TY * TX;
            const int vid = tid + stride; /* vector id to load */
            const int aLoadRow = vid / (BK/4);
            const int aLoadCol = vid % (BK/4);

            AsT[buf][aLoadCol*4][aLoadRow]      = vA[i].x;
            AsT[buf][aLoadCol*4 + 1][aLoadRow]  = vA[i].y;
            AsT[buf][aLoadCol*4 + 2][aLoadRow]  = vA[i].z;
            AsT[buf][aLoadCol*4 + 3][aLoadRow]  = vA[i].w;
        }

#pragma unroll
        for (int i = 0; i < bLoadNum; i++) {
            const int stride = i * TY * TX;
            const int vid = tid + stride; /* vector id to load */
            const int bLoadRow = vid / (BN/4);
            const int bLoadCol = vid % (BN/4);
            *reinterpret_cast<float4*>(&Bs[buf][bLoadRow][bLoadCol*4]) = vB[i];
        }

        __syncthreads();
    }

    // epilogue (last compute phase)
#pragma unroll
    for (int k = 0; k < BK; k++) {
        // load A from AsT to register a
#pragma unroll
        for (int vecIdx = 0; vecIdx < TM/4; vecIdx++) {
            float4 vecA = *reinterpret_cast<float4*>(&AsT[buf][k][warpRow*warpRowSize + vecIdx*16 + laneRow*4]); /* "4" is the number of element to load at a time */

            a[vecIdx*4]      = vecA.x;
            a[vecIdx*4 + 1]  = vecA.y;
            a[vecIdx*4 + 2]  = vecA.z;
            a[vecIdx*4 + 3]  = vecA.w;
        }
        
        // load B from Bs to register b
#pragma unroll
        for (int vecIdx = 0; vecIdx < TN/4; vecIdx++) {
            float4 vecB = *reinterpret_cast<float4*>(&Bs[buf][k][warpCol*warpColSize + vecIdx*32 + laneCol*4]); /* "4" is the number of element to load at a time */
            
            b[vecIdx*4]      = vecB.x;
            b[vecIdx*4 + 1]  = vecB.y;
            b[vecIdx*4 + 2]  = vecB.z;
            b[vecIdx*4 + 3]  = vecB.w;
        }

#pragma unroll
        for (int i = 0; i < TM; i++)
#pragma unroll
            for (int j = 0; j < TN; j++) 
                acc[i][j] += a[i] * b[j];
    }

    // Stores follow the compute mapping, not the load mapping.
    const int threadRow = blockRow + warpRow*warpRowSize + laneRow*4;
    const int threadCol = blockCol + warpCol*warpColSize + laneCol*4;
    int outRow, outCol;
    
#pragma unroll
    for (int i = 0; i < TM; i++) {
        // i % 4 is a position of element in a vector
        // i / 4 is an index of a vector
        outRow = threadRow + (i % 4) + (i / 4)*16;

#pragma unroll
        for (int j = 0; j < TN; j++) {
            // j % 4 is a position of element in a vector
            // j / 4 is an index of a vector
            outCol = threadCol + (j % 4) + (j / 4)*32;

            gemmStoreIfInside(C, N, outRow, outCol, M, N, acc[i][j]);
        }
    }
}

void initMat(int row, int col, float* mat) {
    for (int r = 0; r < row; r++) {
        for (int c = 0; c < col; c++) {
            mat[r * col + c] = (float)(rand() % 10);
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

    dim3 threadsPerBlock(TX, TY);
    dim3 blocksPerGrid((N + BN - 1) / BN, (M + BM - 1) / BM);

    GemmReport rep("sgemm", M, N, K, REPS);
    rep.setLaunch(blocksPerGrid, threadsPerBlock);
    int variant = rep.addVariant("sgemm");

    for (int i = 0; i <= REPS; i++) {
        CHECK_CUDA_ERROR(cudaEventRecord(kernelStart));
        sgemm<<<blocksPerGrid, threadsPerBlock>>>(M, N, K, d_a, d_b, d_c);
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