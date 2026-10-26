# histogram

Counts 2^28 `uint8` values into a 256-bin `unsigned int` histogram.

| Variant | How |
| --- | --- |
| global | one global `atomicAdd` per element |
| shared | per-block copy of the 256 bins in shared memory (1 KB), 492 blocks with a grid-stride loop, merged with 256 global atomics per block |

## Run

```bash
make ARCH=sm_86
./histogram 268435456       # N, number of bytes
./histogram 268435456 --csv
./run.sh                    # reproduces the Result table
```

All 256 bins are checked against the CPU.

## Result

RTX 3090, N = 2^28 (256 MB), kernel time only, 1 warm-up + 10 runs.

GB/s = N bytes ÷ kernel time.

| Variant | ms | GB/s | vs global | Note |
| --- | ---: | ---: | ---: | --- |
| global | 73.7 | 3.64 | 1.00 | 2^28 atomics land on only 256 addresses in L2 and queue up |
| shared | 0.735 | 365 | 100.31 | atomics go to a private copy, only 256 global atomics per block |

Shared reaches 44 % of the 824 GB/s sequential read
([`memory-access/stride`](../../memory-access/stride)): each thread loads one byte,
then waits on the atomic before the next load, so few bytes are in flight.
The 492 blocks are 82 SMs×6, fixed in the source for the RTX 3090.
