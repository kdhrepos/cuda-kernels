#pragma once

// Indexing helpers for the tiled GEMM kernels.
//
// Deliberately small. The tile loops, the shared-memory layouts and the index
// swaps that transpose a tile are what these benchmarks exist to show, so they
// stay written out in each kernel. What lives here is only the part that was
// being retyped -- and mistyped -- in every file.
//
// The naming convention each kernel follows:
//
//   blockRow / blockCol   first row/col of C this block owns
//   kTile                 first k of this step of the K loop
//   loadRow / loadCol     what THIS thread reads during the load phase; rows
//                         walk with stride TILE so a warp's reads coalesce
//   threadRow / threadCol first row/col of the contiguous TM x TN block of C
//                         this thread accumulates and finally stores
//
// loadRow and threadRow are different mappings of the same thread on purpose:
// strided for the load, contiguous for the register tile. Giving them separate
// names is the point -- writing one where the other belongs was a real bug here
// (an A load stepping by the column tile TN instead of the row tile TM, which
// only worked while TM == TN).

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstddef>

// Row-major element, or 0 when the index falls outside the matrix. Folding the
// bounds check and the index into one call means the two cannot drift apart,
// which is how a guard ends up testing a different element than the load reads.
// `ld` is the row stride, normally the matrix's column count.
__device__ inline float gemmLoadOrZero(const float* src, int ld,
                                       int row, int col, int rows, int cols)
{
    if (row >= rows || col >= cols) return 0.0f;
    return src[(size_t)row * ld + col];
}

// The same, for the half-precision inputs the tensor-core kernel takes.
__device__ inline __half gemmLoadOrZero(const __half* src, int ld,
                                        int row, int col, int rows, int cols)
{
    if (row >= rows || col >= cols) return __float2half(0.0f);
    return src[(size_t)row * ld + col];
}

// Row-major store, skipped when the index falls outside the matrix.
__device__ inline void gemmStoreIfInside(float* dst, int ld,
                                         int row, int col, int rows, int cols,
                                         float value)
{
    if (row >= rows || col >= cols) return;
    dst[(size_t)row * ld + col] = value;
}

// Tiles needed to cover `extent` in steps of `tile`.
__host__ __device__ inline int gemmTiles(int extent, int tile)
{
    return (extent + tile - 1) / tile;
}
