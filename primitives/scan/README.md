# scan

Inclusive prefix sum of 2^28 `int` (values 0–9), in three stages: each block
scans its own 256 elements, the block sums are scanned recursively, and every
block then adds the sum of all blocks before it.

| Stage | How |
| --- | --- |
| block-scan | 256-thread Kogge-Stone scan in shared memory, writes each block's sum |
| recursive | the same scan applied to the 2^20 block sums |
| add-sums | adds the preceding block sums to every element |

## Run

```bash
make ARCH=sm_86
./scan 268435456            # N, number of ints
./scan 268435456 --csv
./run.sh                    # reproduces the Result table
```

The full scan is checked against the CPU (on the total row).

## Result

RTX 3090, N = 2^28 (1 GB), kernel time per stage, 1 warm-up + 10 runs.

GB/s = bytes the stage reads and writes ÷ its time.

| Stage | ms | GB/s | % of total | Note |
| --- | ---: | ---: | ---: | --- |
| block-scan | 4.23 | 507 | 62.2 | one load per thread, then 8 steps with 16 barriers and no loads in flight |
| recursive | 0.0341 | 246 | 0.5 | works on 2^20 block sums, negligible |
| add-sums | 2.53 | 847 | 37.3 | pure streaming, saturates bandwidth |
| total | 6.80 | 633 | 100 | |
