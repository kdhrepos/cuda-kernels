# stride

Copies a `float` array visiting every element exactly once, but neighbouring
threads touch elements `stride` apart (stride 1 … 1024, swept inside the program).
One pass makes only the reads strided, the other only the writes.

| Variant | How |
| --- | --- |
| read | strided read, contiguous write |
| write | contiguous read, strided write |

## Run

```bash
make ARCH=sm_86
./stride 16777216           # N, number of floats
./stride 16777216 --csv
./run.sh                    # reproduces the Result table
```

There is no correctness check. Every element is still read and written once.

## Result

RTX 3090, N = 2^24 (64 MB), kernel time only, 1 warm-up + 10 runs.
GB/s = 2 × N × 4 bytes ÷ kernel time (useful bytes, not DRAM traffic).

| Stride | read GB/s | vs stride 1 | write GB/s | vs stride 1 | Note |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 824 | 1.00 | 824 | 1.00 | a warp reads 128 contiguous bytes, 4 full sectors |
| 2 | 556 | 0.67 | 496 | 0.60 | a warp touches up to 32 sectors and uses 4 bytes of each; DRAM itself still moves ~830–860 GB/s |
| 4 | 344 | 0.42 | 285 | 0.35 | |
| 8 | 192 | 0.23 | 153 | 0.19 | |
| 16 | 192 | 0.23 | 133 | 0.16 | |
| 32 | 191 | 0.23 | 95.7 | 0.12 | |
| 64 | 167 | 0.20 | 69.1 | 0.08 | |
| 128 | 123 | 0.15 | 65.8 | 0.08 | scattered writes cost about twice scattered reads |
| 256 | 95.5 | 0.12 | 67.0 | 0.08 | |
| 512 | 331 | 0.40 | 147 | 0.18 | the same lines come back while still in L2; how much depends on N ÷ stride |
| 1024 | 326 | 0.40 | 272 | 0.33 | |

At N = 2^26 (256 MB) the recovery at large strides is gone: stride 512 and 1024 reads
reach 65.8 and 82.5 GB/s (vs 331 and 326 at 2^24), while stride 1 reads 854 GB/s.
The stride 1 read at N = 2^24 (824 GB/s) is the bandwidth other folders compare against.
