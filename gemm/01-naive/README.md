# 01-naive

`float` matrix multiply C = A × B (row-major), one thread per element of C, every
operand read straight from global memory. `naive_gemm_swapped` is the same kernel
but the thead block's x mapped to rows of C instead of columns.

| Variant | How |
| --- | --- |
| naive | `threadIdx.x` → column of C, `threadIdx.y` → row |
| swapped | `threadIdx.x` → row of C, `threadIdx.y` → column |

## Run

```bash
make ARCH=sm_86
./naive_gemm 16 16 4096 4096 4096          # bx by M N K
./naive_gemm 16 16 4096 4096 4096 --csv
./run.sh                                   # reproduces the Result table
```

The top-left 1024 × 1024 corner of C is checked against the CPU.

## Result

RTX 3090, M×N×K = 4096×4096×4096, kernel time only, 1 warm-up + 10 runs.

GFLOPS = 2×M×N×K ÷ (kernel time).

| Block (bx × by) | naive GFLOPS | vs 16x16 | swapped GFLOPS | naive ÷ swapped | Why |
| --- | ---: | ---: | ---: | ---: | --- |
| 1x256 | 299 | 0.13 | 1,210 | 0.25 | a warp reads less than a full 32-byte sector per row |
| 2x128 | 592 | 0.26 | 2,220 | 0.27 | a warp reads less than a full 32-byte sector per row |
| 4x64 | 1,090 | 0.48 | 1,860 | 0.58 | a warp reads less than a full 32-byte sector per row |
| 8x32 | 1,860 | 0.82 | 1,100 | 1.69 | |
| 16x16 | 2,260 | 1.00 | 588 | 3.84 | per k step a block needs bx + by distinct values (16 + 16 = 32); 256x1 needs 1 + 256 = 257 for the same work |
| 32x8 | 2,250 | 0.99 | 280 | 8.02 | swapped: neighbouring threads read different rows of A, K × 4 bytes apart |
| 64x4 | 1,740 | 0.77 | 246 | 7.06 | |
| 128x2 | 918 | 0.41 | 116 | 7.92 | |
| 256x1 | 515 | 0.23 | 50.9 | 10.13 | |
| 8x8 | 1,830 | 0.81 | 966 | 1.90 | |
| 32x32 | 2,080 | 0.92 | 303 | 6.84 | |

With 16x16 blocks the rate stays flat across sizes (2,170 – 2,240 GFLOPS from
1000 × 1100 × 900 to 4000 × 4100 × 3900), because the kernel is memory-bound at every size.

Reading the table: a warp is 32 consecutive `threadIdx.x` values, so `bx` decides what one
warp touches. In naive, x walks along a row of C, and only 16x16 and 32x8 reach
~2,250 GFLOPS. Below bx = 8 the rate falls fast (1x256: 299, 13 % of 16x16) because a warp
reads partial sectors of B; above bx = 32 it falls again (256x1: 515) even though every warp
is fully coalesced, because the block covers fewer rows and needs more distinct values per
k step (1 + 256 = 257 at 256x1). Swapped flips the picture: x walks down a column, so it is
best when bx is small (2x128: 2,220 GFLOPS, where a warp is 2 rows × 16 adjacent columns,
the same footprint as a naive 16x16 warp) and worst at 256x1 (50.9 GFLOPS, ×10 slower than
naive), where the 32 threads of a warp sit in one column and read 32 different rows of A,
K × 4 = 16 KB apart.
