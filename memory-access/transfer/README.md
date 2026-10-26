# transfer

`int` vector add, N = 2^26 (256 MB per array): 512 MB goes to the GPU and 256 MB
comes back. The same work is done five ways that differ only in how the data
moves between host and device.

| Variant | How |
| --- | --- |
| pageable | `malloc` buffers, `cudaMemcpy` |
| pinned | `cudaMallocHost` buffers, `cudaMemcpy` |
| managed | `cudaMallocManaged`, pages migrate on GPU page faults |
| prefetch | managed + `cudaMemPrefetchAsync` before the kernel |
| streams | pinned, 8 chunks on 8 streams so copies overlap the kernel |

## Run

```bash
make ARCH=sm_86
./transfer 26               # power, N = 2^power
./transfer 26 --csv         # summary only; per-stage times are in the plain output
./run.sh                    # reproduces the Result table
```

Every element of c is checked against the CPU.

## Result

RTX 3090, N = 2^26, end-to-end time (copies + kernel), 1 warm-up + 10 runs.

vs pinned = pinned time ÷ variant time.

| Variant | total ms | vs pinned | Note |
| --- | ---: | ---: | --- |
| pageable | 61.7 | 0.52 | the driver first copies through a pinned staging buffer and then the GPU memory |
| pinned | 32.2 | 1.00 | the copy engine reads the user buffer directly |
| managed | 46.9 | 0.69 | the GPU stalls on page faults that the CPU driver resolves in batches |
| prefetch | 33.5 | 0.96 | one bulk migration, close to pinned |
| streams | 28.0 | 1.15 | copies overlap the kernel, but both directions share a host-side path, so only ×1.15 |