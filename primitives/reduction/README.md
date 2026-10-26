# reduction

Sums a float array. Each 256-thread block reduces its 256 elements to one partial
sum and writes it to `in[block * 256]`. The host adds the partial sums.

The three kernels differ only in how a block reduces its 256 values:

| Variant | How |
| --- | --- |
| interleaved | shared memory tree, stride 1 → 128, thread `tx` adds if `tx % 2s == 0` |
| sequential | shared memory tree, stride 128 → 1, thread `tx` adds if `tx < s` |
| warp-shuffle | `__shfl_down_sync` within each warp, then once more across the 8 warp sums |

## Run

```bash
make ARCH=sm_86
./reduction 67108864        # N, number of floats
./reduction 67108864 --csv
./run.sh                    # reproduces the Result table
```

Each block's partial sum is checked against the CPU.

## Result

RTX 3090, N = 2^26 (268 MB), kernel time only, 1 warm-up + 10 runs.

GB/s = (N × 4 bytes) ÷ kernel time.

| Variant | GB/s | × interleaved | Note |
| --- | ---: | ---: | --- |
| interleaved | 201 | 1.00 | active threads spread over all 8 warps, so every warp keeps issuing |
| sequential | 326 | 1.62 | active threads packed into the first warps |
| warp-shuffle | 636 | 3.17 | 9 block barriers → 1, so blocks finish sooner and new loads start |

For reference, a plain sequential read on this GPU reaches 824 GB/s
([`memory-access/stride`](../../memory-access/stride)). Warp-shuffle stops at 77 %
of that because each thread issues only one 4-byte load.
