# vecadd

Adds two `int` vectors, c = a + b, with n = 2^26 elements (256 MB per vector).
The host→device copy, the kernel and the device→host copy are timed separately,
so the table shows where the end-to-end time goes.

## Run

```bash
make ARCH=sm_86
./vecadd 67108864           # n, number of ints
./vecadd 67108864 --csv     # output format as csv
./run.sh                    # reproduces the Result table below
```

Every element of c is checked against the CPU.

## Result

RTX 3090, n = 2^26, each stage timed on its own, 1 warm-up + 10 runs.
GB/s = bytes moved by the stage ÷ its time.

| Stage | ms | GB/s | % of total |
| --- | ---: | ---: | ---: |
| H2D | 38.2 | 14.1 | 62.4 |
| kernel | 0.950 | 847 | 1.6 |
| D2H | 22.1 | 12.1 | 36.1 |
| total | 61.3 | | 100 |

PCIe moves data far slower than the GPU's own DRAM, so the H2D and D2H data copies dominate the total execution time. As a result, the two copies take 98 % of the total.