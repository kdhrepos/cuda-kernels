# p-chase

This kernel measures load latency. In the kernel, only one thread is executed. It follows a chain `j = arr[j]` laid out in 128-byte steps, 2^18 dependent loads, and reports cycles per load. The array
grows from 4 KB to 64 MB (swept inside the program), so it walks through L1 cache,
L2 cache and global memory.

## Run

```bash
make ARCH=sm_86
./p_chase                   # no arguments
./p_chase --csv
./run.sh                    # reproduces the Result table
```

There is no correctness check.

## Result

RTX 3090, one thread, one run per size, timed with `clock64()` inside the kernel.

cycles/load = elapsed SM cycles ÷ 2^18 loads.

| Array size | cycles/load | Notes |
| ---: | ---: | --- |
| 4 – 64 KB | 38.9 | each load depends on the previous one, so the time per load is the full latency of the level that the array fits in |
| 128 KB | 154 | only part of the array fits in L1 |
| 256 KB – 2 MB | 256.5 | |
| 4 MB | 256 | |
| 8 – 64 MB | 525 | |