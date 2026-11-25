# cuda-kernels

CUDA kernels written from scratch, each with a small benchmark that reports time
and throughput. (Draft)

## Environment

Results in this repository were measured on:

| | |
|---|---|
| GPU | 2× NVIDIA GeForce RTX 3090 (compute capability 8.6, `sm_86`) |
| CUDA toolkit | 12.3 (nvcc 12.3.52) |
| NVIDIA driver | 545.23.06 |
| Host compiler | gcc 9.4.0 |
| Build | GNU Make 4.2.1, `-O3 -std=c++14 -arch=sm_86` |
| Python | 3.8.10 |

## Layout

| Directory | Contents |
|---|---|
| `intro/` | device-query, divergence, matadd, vecadd |
| `gemm/` | multiple versions of GEMM optimization compared to cuBLAS |
| `memory-access/` | p-chase, stride, transfer, transpose |
| `primitives/` | parallel programming algorithms such as histogram, reduction, scan |
| `llm-inference/` | decoder, graph-launch, normalization, quantized-gemv |
| `common/` | shared headers and build settings (`common.mk`); see [common/README.md](common/README.md) |
| `tools/` | runs benchmarks over a set of sizes and collects the results |

## Build

```bash
make                 # build everything
make gemm            # build one group
make ARCH=sm_86      # target architecture (default sm_80; the RTX 3090 needs sm_86)
make clean
```

A single benchmark is quickest built in its own directory with `make`.

## Run

Every binary prints one CSV row per variant when given `--csv`:
`name,variant,m,n,k,iter,avg_ms,min_ms,max_ms,gflops,status`

```bash
./gemm/02-tiled/tiled_gemm 2048 2048 2048          # M N K
./gemm/01-naive/naive_gemm 16 16 2048 2048 2048    # bx by M N K
```

## Collecting results

Each benchmark folder has a `run.sh` that runs its binaries with `--csv` and
the arguments to measure. Edit it to change sizes or shapes.
`tools/sweep.py` runs them and saves the output as
`results/<group>/<kernel>/<kernel>.csv`.

```bash
tools/sweep.py                    # every run.sh
tools/sweep.py gemm               # one group
tools/sweep.py gemm/02-tiled      # one kernel
```