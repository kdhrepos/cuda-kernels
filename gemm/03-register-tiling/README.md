# 03-register-tiling

`float` matrix multiply C = A × B, 4096³. Builds on the shared-memory tiles of
[`02-tiled`](../02-tiled), but each thread keeps several elements of C in registers,
so data can be reused on the register, not only shared memory.

| Variant | How |
| --- | --- |
| 1x4-strided | 1×4 outputs per thread, B read with scalar loads |
| 1x4-contig | 1×4 adjacent outputs per thread, B read as one 128-bit load |
| 4x4 | 4×4 outputs per thread, 64×64 block tile, outer product |
| 4x4-transA | 4x4 with the A tile stored transposed in shared memory |
| 4x4-transA-pad | 4x4-transA with a padded A tile |

## Run

```bash
make ARCH=sm_86
./thread_tile_4x4 4096 4096 4096           # M N K, same for every binary
./thread_tile_4x4 4096 4096 4096 --csv
./run.sh                                   # reproduces the Result table
```

The top-left 1024 × 1024 corner of C is checked against the CPU.

## Result

RTX 3090, 4096³, kernel execution time only, 1 warm-up + 10 runs.

This table is the average. GFLOPS = 2×M×N×K ÷ (kernel time).

| Variant | GFLOPS | vs tiled (TILE 16) | Note |
| --- | ---: | ---: | --- |
| tiled, TILE 16 | 2,950 | 1.00 | |
| 1x4-strided | 3,820 | 1.29 | one A value feeds 4 outputs, but B is read with scalar loads |
| 1x4-contig | 5,470 | 1.85 | 4 adjacent columns load as one 128-bit shared read |
| 4x4 | 13,400 | 4.55 | the outer product reuses both A and B values, 0.5 shared values per FMA |
| 4x4-transA | 11,000 | 3.74 | storing A transposed causes a 16-way bank conflict on the shared store |
| 4x4-transA-pad | 14,000 | 4.75 | padding removes the conflict and uses fewer registers (48), so occupancy rises to 81 % |

4x4-transA-pad reaches 39% of the FP32 HW peak performance (82 SMs × 128 FMA × 2 × 1.695 GHz ≈ 35.6 TFLOPS).
