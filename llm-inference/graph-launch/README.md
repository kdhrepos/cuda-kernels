# graph-launch

Launches a tiny kernel (+1 on a vector `float[1024]`, 4 blocks × 256 threads) 100 times on
one stream, normally and as a CUDA Graph captured once and replayed. The kernel
does almost nothing, so most of the time is kernel launch overhead.

| Variant | How |
| --- | --- |
| normal | 100 kernel launches per iteration |
| graph | the 100 kernel launches captured once, instantiated once, replayed with `cudaGraphLaunch` |

## Run

```bash
make ARCH=sm_86
./graph_launch              # no arguments
./graph_launch --csv
./run.sh                    # reproduces the Result table
```

There is no correctness check.

## Result

RTX 3090, 100 kernels per iteration, 10 timed iterations after a separate warm-up.
- dispatch = until the launch calls return. 
- end-to-end = until the work is done.

Times are per kernel.

| Variant | dispatch µs | end-to-end µs | vs normal (end-to-end) | Why |
| --- | ---: | ---: | ---: | --- |
| normal | 1.1 | 1.8 | 1.00 | End-to-end > dispatch: the GPU, not the CPU, is the bottleneck. It spends time between kernels |
| graph | 0.010 | 1.1 | 1.70 | The GPU knows the whole sequence ahead, so the gap between kernels shrinks |