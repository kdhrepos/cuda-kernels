# cuda-kernels

CUDA kernels written from scratch, each with a small benchmark that reports time
and throughput. (Draft)

## Layout

| Directory | Contents |
|---|---|
| `intro/` | device-query, divergence, matadd, vecadd |
| `gemm/` | naive, tiled, register-tiling and tensor-core GEMM, raced against cuBLAS |
| `memory-access/` | p-chase, stride, transfer, transpose |
| `primitives/` | histogram, reduction, scan |
| `llm-inference/` | decoder, graph-launch, normalization, quantized-gemv |
| `common/` | shared headers and build settings (`common.mk`); see [common/README.md](common/README.md) |
| `tools/` | `sweep.py`, runs benchmarks over a set of sizes and collects the results |

## Build

```bash
make                 # build everything
make gemm            # build one group
make ARCH=sm_86      # target another architecture (default sm_80)
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


## TODO

- GEMM performance table per version
