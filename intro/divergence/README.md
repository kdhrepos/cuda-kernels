# divergence

Kernels that do the same FMA work (n = 2^22 threads, 5000 FMAs each) but each thread walks through difference code block to show the cost of warp divergence.

| Variant | How |
| --- | --- |
| thread-divergence | branch on `tid % 2`: divergence **inside the warp** |
| warp-divergence | branch on `(tid / 32) % 2`: divergence **between the warps** |
| k-divergence | k-way `switch` on the thread index |
| vecadd with branch | memory-bound add with a branch |

## Run

```bash
make ARCH=sm_86
./divergence 4 4194304 5000          # k, n, FMA loop iterations
./divergence 4 4194304 5000 --csv
./run.sh                             # reproduces the Result table (k = 1 … 32)
```

## Result

RTX 3090, n = 2^22, 5000 FMAs per thread, kernel time only, 1 warm-up + 10 runs.

vs warp-div = (kernel time) ÷ (warp-div time) (k = 1 run, 1.42 ms); higher is slower.

| Variant | ms | vs warp-div | Why |
| --- | ---: | ---: | --- |
| warp-div | 1.42 | 1.00 | same work as thread-div; splitting between warps runs one path |
| thread-div | 2.83 | 1.99 | same work as warp-div; splitting inside a warp runs both paths |
| vecadd | 0.0637 | 0.04 | memory-bound, and the compiler turns the short branch into predicated instructions, so the extra work hides behind memory latency |
| k-div, k = 1 | 1.43 | 1.01 | a warp runs each distinct path in turn, so time grows with k |
| k-div, k = 2 | 2.84 | 2.00 | |
| k-div, k = 4 | 5.38 | 3.79 | |
| k-div, k = 8 | 10.2 | 7.20 | |
| k-div, k = 16 | 20.6 | 14.52 | |
| k-div, k = 32 | 39.9 | 28.10 | |

Relative to k-div at k = 1, the k-way times are ×1.00 / 1.99 / 3.77 / 7.15 / 14.43 / 27.93.