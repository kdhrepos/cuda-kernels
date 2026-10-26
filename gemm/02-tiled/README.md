# 02-tiled

`float` matrix multiply C = A × B, 4096³. Each block loads TILE × TILE tiles of A
and B into shared memory and every thread reuses them for TILE multiply-adds
before the next tile. The value of TILE is 16 in default.

## Run

```bash
make ARCH=sm_86                          # TILE 16; make DEFINES='TILE=32' for another
./tiled_gemm 4096 4096 4096              # M N K
./tiled_gemm 4096 4096 4096 --csv
./run.sh                                 # reproduces the Result table
```

The top-left 1024×1024 corner of C is checked against the CPU.

## Result

RTX 3090, 4096³, kernel time only, 1 warm-up + 10 runs.

GFLOPS = 2×M×N×K ÷ (kernel time).

| TILE | GFLOPS | vs TILE 16 | Note |
| --- | :---: | --- | --- |
| 8 | 1,870 | 0.63 | each loaded value is reused fewer times |
| 16 | 2,950 | 1.00 | limited by shared-memory loads, about 1.25 per FMA (Nsight: L1/TEX 97.8 %, top stall MIO throttle), not by DRAM (33 %) |
| 32 | 2,720 | 0.92 | larger blocks fit fewer per SM (8 vs 12 warps per scheduler), so less latency is hidden |

TILE 16 is ×1.30 over the naive kernel with 16x16 blocks at the same size
(2,260 GFLOPS, [`01-naive`](../01-naive)).
