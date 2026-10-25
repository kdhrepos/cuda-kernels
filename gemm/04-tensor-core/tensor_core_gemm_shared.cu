#include "../../common/cuda_check.cuh"
#include "../../common/gemm_report.cuh"

#include <iostream>
#include <string>
#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>
#include <mma.h>        // WMMA
#include <cuda_fp16.h>  // __half, __float2half
#include <cuda_bf16.h>  // __nv_bfloat16, __float2bfloat16

// Measured iterations, after one discarded warm-up. Override at build time with
// -DREPS=n.
#ifndef REPS
#define REPS 10
#endif

using namespace nvcuda;



#ifndef TILE
#define TILE 16
#endif

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


__global__ void tensorCoreGemm(
    const int M,
    const int N,
    const int K,
    const __half *A,
    const __half *B,
    float *C
) {
    __shared__ __align__(32) __half As[32][16+8], Bs[16][32+8]; // padding +8 half
    __shared__ float Cs[4][16][16];   // staging tile for partial tiles, one per warp
    int warpId = threadIdx.x / 32;
    int wr = warpId/2, wc = warpId%2;

    int tid = threadIdx.x;
    int by = blockIdx.y;
    int bx = blockIdx.x;

    wmma::fragment<wmma::accumulator, 16,16,16, float> c;
    wmma::fill_fragment(c, 0.0f);
    for (int p = 0; p < (16+K-1)/16; ++p) {
        for (int j=0; j<4; j++) {
            int r0 = tid/16 + j*8; // position of a warp in the tile A (row)
            int aRow = r0 + by*32, aCol = tid%16 + p*16; // position in the thread block 
            As[r0][tid%16] = ((aRow < M) && (aCol < K)) ? A[(aRow)*K + aCol] : __float2half(0.0f);
            
            int r1 = tid/32 + j*4; // position of a warp in the tile B (row)
            int bRow = r1 + p*16, bCol = tid%32 + bx*32; // position in the thread block
            Bs[r1][tid%32] = ((bRow < K) && (bCol < N)) ? B[(bRow)*N + bCol] : __float2half(0.0f);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16,16,16, __half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16,16,16, __half, wmma::row_major> b;
        wmma::load_matrix_sync(a, &As[wr*16][0], 16+8);
        wmma::load_matrix_sync(b, &Bs[0][wc*16], 32+8);
        wmma::mma_sync(c, a, b, c);
        __syncthreads();
    }

    int cRow = by*32 + wr*16;
    int cCol = bx*32 + wc*16;

    // Storing straight into C needs two things of N. The tile must fit, because
    // store_matrix_sync writes the full 16 columns of every row and anything past
    // column N would land in the next row of C. And N, the leading dimension, must
    // be a multiple of 16 bytes -- 4 floats -- or the store is misaligned and
    // writes garbage from its second row on.
    if (cRow + 16 <= M && cCol + 16 <= N && N % 4 == 0) {
        wmma::store_matrix_sync(&C[cRow*N + cCol], c, N, wmma::mem_row_major);
    } else {
        // Otherwise stage the tile at ldm = 16 (64 bytes, properly aligned) and
        // copy back only the part that belongs to C.
        wmma::store_matrix_sync(&Cs[warpId][0][0], c, 16, wmma::mem_row_major);
        __syncwarp();

        int lane = threadIdx.x % 32;
        for (int idx = lane; idx < 16*16; idx += 32) {
            int r = idx / 16;
            int cc = idx % 16;
            if (cRow + r < M && cCol + cc < N)
                C[(cRow + r)*N + cCol + cc] = Cs[warpId][r][cc];
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

    float *h_a = (float*)malloc(sizeof(float) * M * K);
    float *h_b = (float*)malloc(sizeof(float) * K * N);
    float *h_c = (float*)malloc(sizeof(float) * M * N);

    initMat(M, K, h_a);
    initMat(K, N, h_b);

    __half *h_a16 = (__half*)malloc(sizeof(__half) * M * K);
    __half *h_b16 = (__half*)malloc(sizeof(__half) * K * N);
    for (size_t i = 0; i < (size_t)M * K; ++i) h_a16[i] = __float2half(h_a[i]);
    for (size_t i = 0; i < (size_t)K * N; ++i) h_b16[i] = __float2half(h_b[i]);
    
    __half *d_a, *d_b;
    float  *d_c;

    CHECK_CUDA_ERROR(cudaMalloc(&d_a, sizeof(__half) * M * K));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, sizeof(__half) * K * N));
    CHECK_CUDA_ERROR(cudaMalloc(&d_c, sizeof(float)  * M * N));
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a16, sizeof(__half) * M * K, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b16, sizeof(__half) * K * N, cudaMemcpyHostToDevice));

    cudaEvent_t kernelStart, kernelEnd;
    CHECK_CUDA_ERROR(cudaEventCreate(&kernelStart));
    CHECK_CUDA_ERROR(cudaEventCreate(&kernelEnd));


    dim3 threadsPerBlock(128);
    dim3 blocksPerGrid((32+N-1)/32, (32+M-1)/32);

    GemmReport rep("tensor_core_gemm", M, N, K, REPS);
    rep.setLaunch(blocksPerGrid, threadsPerBlock);
    rep.setElementBytes(sizeof(__half), sizeof(float));   // half in, float accumulate
    int wmmaVariant = rep.addVariant("wmma");

    for (int i=0; i<=REPS; i++) {
        CHECK_CUDA_ERROR(cudaEventRecord(kernelStart));
        tensorCoreGemm<<<blocksPerGrid, threadsPerBlock>>>(M, N, K, d_a, d_b, d_c);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaEventRecord(kernelEnd));
        CHECK_CUDA_ERROR(cudaEventSynchronize(kernelEnd));

        float kernelTime=0.0;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernelTime, kernelStart, kernelEnd));
        // i == 0 is the warm-up: verified, then thrown away.
        if (i==0) {
            CHECK_CUDA_ERROR(cudaMemcpy(h_c, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
            rep.setCheck(wmmaVariant, gemmCheckCpu(M, N, K, h_a, h_b, h_c));
        } else {
            rep.addSample(wmmaVariant, kernelTime);
        }
    }

    rep.print(csv);

    CHECK_CUDA_ERROR(cudaEventDestroy(kernelStart));
    CHECK_CUDA_ERROR(cudaEventDestroy(kernelEnd));

    free(h_a);
    free(h_b);
    free(h_c);
    free(h_a16); 
    free(h_b16);

    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_c));

    return rep.passed() ? 0 : 1;
}