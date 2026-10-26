# cuda-kernels

CUDA kernels written from scratch, each with a small benchmark that reports time
and throughput. (Draft)

## Layout

| Directory | Contents |
|---|---|
| `intro/` | device-query, divergence, matadd, vecadd |
| `gemm/` | naive, tiled, register-tiling and tensor-core GEMM, raced against cuBLAS |
| `memory-access/` | chase, stride, transfer, transpose |
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

## Sweeping sizes

```bash
tools/sweep.py -s 512,1024,2048,4096 -a '{s} {s} {s}' --out out.csv gemm/02-tiled/tiled_gemm
```

- `-a`: argument template; `{s}` is replaced by each size
- `-m`: metric to pivot into the printed table (`gflops`, `avg_ms`, ...)
- `--out`: also write every raw CSV row

Block-shape comparison for the naive GEMM: `gemm/01-naive/sweep_block.sh`.

## TODO

- GEMM performance table per version
- Fill in per-directory descriptions
