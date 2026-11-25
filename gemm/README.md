# gemm

`float` matrix multiply C = A × B (row-major), written in steps from a naive kernel to a
warp-tiled one and compared with cuBLAS.

| Directory | How |
| --- | --- |
| [01-naive](01-naive) | one thread per element of C, every operand read from global memory |
| [02-tiled](02-tiled) | TILE × TILE tiles of A and B in shared memory |
| [03-register-tiling-4x4](03-register-tiling-4x4) | each thread keeps several outputs in registers |
| [04-register-tiling-8x8](04-register-tiling-8x8) | 8×8 outputs per thread, 128×128 block tile |
| [05-double-buffering](05-double-buffering) | the next tile is loaded while the current one is computed |
| [06-warp-tiling](06-warp-tiling) | the block tile is split over warps, 4×8 lanes each |
| [cublas-gemm](cublas-gemm) | cuBLAS SGEMM, the reference |

## Run

```bash
cd 02-tiled
make ARCH=sm_86
./tiled_gemm 4096 4096 4096          # M N K
./tiled_gemm 4096 4096 4096 --csv
```

01-naive takes the block shape first (`bx by M N K`). `make gemm` from the repository root
does not build 05 and 06, so run `make ARCH=sm_86` inside those directories.
The top-left 1024 × 1024 corner of C is checked against the CPU.

## Result

RTX 3090, 4096³, kernel time only.

GFLOPS = 2×M×N×K ÷ (kernel time).

| Version | GFLOPS | vs cuBLAS | Note |
| --- | ---: | ---: | --- |
| 01-naive (16x16) | 2,260 | 0.09 | |
| 02-tiled (TILE 16) | 2,950 | 0.12 | |
| 03-register-tiling-4x4 | 14,000 | 0.58 | 4x4-transA-pad, the fastest of the five variants |
| 04-register-tiling-8x8 | 18,600 | 0.77 | |
| 05-double-buffering | 19,300 | 0.79 | only ×1.04 over 04, and it uses 168 registers against 119 |
| 06-warp-tiling | 20,900 | 0.86 | default 128×128 block tile |
| 06-warp-tiling | 22,000 | 0.91 | 64×128 block tile, the best of 23 swept configs |
| cuBLAS SGEMM | 24,300 | 1.00 | |

01 to 03 are the numbers from their own READMEs. The rest is the mean of two runs.
04 to 06 have no `run.sh` yet.

`06-warp-tiling/sweep/sweep.py` runs 23 configs at 1024, 2048, 4096, 8192 and 16384.
`BM=64 BN=128 TM=8 TN=8 BK=8` is the best at every size, reaching 96 % of cuBLAS at 1024,
93 % at 4096 and 82 % at 16384.
