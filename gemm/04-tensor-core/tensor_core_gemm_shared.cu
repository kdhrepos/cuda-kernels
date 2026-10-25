#include "../../common/cuda_check.cuh"
#include "../../common/gemm_report.cuh"
#include "../../common/gemm_tile.cuh"

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
    // One wmma fragment is FRAG x FRAG; the four warps of this block cover a
    // BLOCK_TILE x BLOCK_TILE square of C as a 2x2 grid of fragments.
    const int FRAG       = 16;
    const int BLOCK_TILE = 32;
    const int WARPS      = 4;

    // The +8 pad keeps a warp's fragment loads off a single bank.
    __shared__ __align__(32) __half As[BLOCK_TILE][FRAG+8], Bs[FRAG][BLOCK_TILE+8];
    __shared__ float Cs[WARPS][FRAG][FRAG];   // staging tile, one per warp

    const int tid    = threadIdx.x;
    const int warpId = tid / 32;
    const int wr = warpId/2, wc = warpId%2;   // this warp's fragment in the 2x2

    // This block owns BLOCK_TILE rows and BLOCK_TILE columns of C.
    const int blockRow = blockIdx.y * BLOCK_TILE;
    const int blockCol = blockIdx.x * BLOCK_TILE;

    wmma::fragment<wmma::accumulator, 16,16,16, float> c;
    wmma::fill_fragment(c, 0.0f);
    for (int p = 0; p < gemmTiles(K, FRAG); ++p) {
        const int kTile = p * FRAG;   // first k of this step

        // Load phase: the whole block cooperates, so a thread's position here
        // has nothing to do with the fragment its warp will compute. Rows are
        // covered in strides (8 for As, 4 for Bs) to keep each read coalesced.
        for (int j=0; j<4; j++) {
            const int asRow = tid/16 + j*8;
            As[asRow][tid%16] =
                gemmLoadOrZero(A, K, blockRow + asRow, kTile + tid%16, M, K);

            const int bsRow = tid/32 + j*4;
            Bs[bsRow][tid%32] =
                gemmLoadOrZero(B, N, kTile + bsRow, blockCol + tid%32, K, N);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16,16,16, __half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16,16,16, __half, wmma::row_major> b;
        // Compute phase: now the warp's own (wr, wc) selects its fragment.
        wmma::load_matrix_sync(a, &As[wr*FRAG][0], FRAG+8);
        wmma::load_matrix_sync(b, &Bs[0][wc*FRAG], BLOCK_TILE+8);
        wmma::mma_sync(c, a, b, c);
        __syncthreads();
    }

    // Where this warp's fragment lands in C.
    const int cRow = blockRow + wr*FRAG;
    const int cCol = blockCol + wc*FRAG;

    // Storing straight into C needs two things of N. The tile must fit, because
    // store_matrix_sync writes the full 16 columns of every row and anything past
    // column N would land in the next row of C. And N, the leading dimension, must
    // be a multiple of 16 bytes -- 4 floats -- or the store is misaligned and
    // writes garbage from its second row on.
    if (cRow + FRAG <= M && cCol + FRAG <= N && N % 4 == 0) {
        wmma::store_matrix_sync(&C[(size_t)cRow*N + cCol], c, N, wmma::mem_row_major);
    } else {
        // Otherwise stage the tile at ldm = FRAG (64 bytes, properly aligned)
        // and copy back only the part that belongs to C.
        wmma::store_matrix_sync(&Cs[warpId][0][0], c, FRAG, wmma::mem_row_major);
        __syncwarp();

        const int lane = tid % 32;
        for (int idx = lane; idx < FRAG*FRAG; idx += 32) {
            const int r  = idx / FRAG;
            const int cc = idx % FRAG;
            gemmStoreIfInside(C, N, cRow + r, cCol + cc, M, N, Cs[warpId][r][cc]);
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