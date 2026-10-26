# decoder

One decode step of a Llama-2-7B-sized layer in FP16. GEMV: W [4096 × 4096] × x →
FP32 y. 

Attention: one new query against a KV cache of 32 heads × 128 dims and
T = 4096 tokens.

| Variant | How |
| --- | --- |
| gemv | one warp per row of W, 16 bytes per lane per load |
| v1 | attention, one block per head (32 blocks) |
| v2 split | attention, each head's context split into 16 chunks (512 thread blocks) with a merge kernel |

## Run

```bash
make ARCH=sm_86
./decoder 4096 16           # T (context length), number of splits for v2
./decoder 4096 16 --csv
./run.sh                    # reproduces the Result table
```

Each output is checked against a CPU reference.

## Result

RTX 3090, T = 4096, kernel time only (v2 includes the merge), 1 warm-up + 10 runs.
GB/s = bytes the kernel must read ÷ kernel time.

| Variant | ms | GB/s | × v1 | Note |
| --- | ---: | ---: | ---: | --- |
| gemv | 0.0416 | 808 | | reaches 98 % of the 824 GB/s sequential read |
| v1 | 0.118 | 571 | 1.00 | 32 thread blocks for 82 SMs, so most of the GPU is idle |
| v2 split | 0.0896 | 749 | 1.31 | 512 thread blocks keep every SM loading, while the merge kernel is small |

Both are ~1 FLOP per byte, far below the ridge point, so memory bandwidth is the only thing
that matters.