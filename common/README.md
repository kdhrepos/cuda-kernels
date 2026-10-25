# common/

Shared headers for the benchmarks in this repo. Header-only — nothing to build,
just include with a relative path. A file two levels down (`gemm/tiled-gemm/`,
`fundamentals/01-vecadd/`) uses `../../common/`; one level down uses `../common/`.
Moving a benchmark to a different depth means fixing that prefix.

## cuda_check.cuh

The error-check helpers every file used to carry its own copy of.

```cpp
#include "../../common/cuda_check.cuh"

CHECK_CUDA_ERROR(cudaMalloc(&d_a, bytes));   // any call returning cudaError_t
CHECK_LAST_CUDA_ERROR();                     // right after a kernel launch
```

Both print file:line plus the failing call to `stderr` and exit. `CHECK_CUDA_LAST_ERROR()`
is an alias for `CHECK_LAST_CUDA_ERROR()`, so either word order compiles.

`WARN_LAST_CUDA_ERROR()` reports and keeps going instead of exiting. 02-matadd and
gray-to-scale want that on purpose, so they use it rather than the exiting macro.

## cublas_check.cuh

`CHECK_CUBLAS(...)`, split out so only the benchmarks that race against cuBLAS
have to link `-lcublas`.

## bench_format.cuh

The pieces of the report format, in `namespace bench`. The left 43 columns are
the same in every benchmark —

```
" " + name(9) + 3 x [ time(8) + " " + unit(2) ]
```

— and each benchmark appends the metric columns it cares about. Build the header
row with `timeHeader()` + `rightLabel()` so labels land on the same column edges
as the data.

| | |
|---|---|
| `Stat`, `summarize()` | avg / min / max over the measured iterations (ms) |
| `Unit`, `pickUnit()` | one unit per table, chosen from the fastest row so no row collapses to `0.03 ms` |
| `gbPerSec()`, `gflopPerSec()` | 10-based GB/s and GFLOP/s from bytes-or-flops and ms |
| `humanBytes()` | `25.0 KiB`, `1.1 GiB` |
| `banner()`, `bannerEnd()`, `rule()`, `configLine()` | table chrome |
| `nameCell()`, `timeCells()` | the shared left columns of a data row |
| `takeFlag()` | pulls `--csv` out of argv so the positional checks stay as written |

## kernel_report.cuh

One row per variant, which is what most benchmarks here are doing: time N kernels
on the same input and compare them.

```cpp
KernelReport rep("reduction", iter);
rep.addConfig("N", std::to_string(N) + " elements");
rep.addIterations();
int interleaved = rep.addVariant("interleaved");
int shuffled    = rep.addVariant("warp-shuffle");
rep.setBytes(interleaved, 4.0*N);       // -> GB/s column
rep.setBytes(shuffled,    4.0*N);
rep.setBaseline(interleaved);           // -> "x base" column
...
rep.addSample(shuffled, kernelTime);
rep.setCheck(shuffled, mismatches);     // -> "check" column
rep.print(csv);
return rep.passed() ? 0 : 1;
```

Columns appear only when they carry something, so a single-variant benchmark gets a
plain table and a sweep gets a wide one.

| | |
|---|---|
| `setBytes` / `setFlop` | adds a GB/s or GFLOP/s column |
| `setBaseline(v)` | adds `x base`, every row's speedup against `v` |
| `setCheck(v, n)` | adds `check`, PASS/FAIL per row |
| `setTotalRow("total")` | sums the rows under the rule -- for stages of one pipeline, not rivals |
| `setTotalCheck(n)` | the verification belongs to the whole pipeline |
| `setVariantHeader("stride")` | renames the first column when "variant" is wrong |
| `setShowMedian(true)` | adds `med`, for harnesses that headline the median |
| `setShowTimes(false)` | no time columns at all, for a benchmark that measures something else |
| `addColumn` / `setCell` | any column the report does not model |

The time column widens itself to the slowest row while the unit still comes from the
fastest, so a table spanning a 50x speedup stays aligned without losing digits. With
one sample per variant there is no spread to show, so avg/min/max collapse into a
single `time` column.

### Columns of your own

Whatever the report does not model gets a column via `addColumn(header, width,
precision, scientific)` and `setCell(variant, column, value)`:

```cpp
int colErr  = rep.addColumn("max rel err", 13, 1, true);   // 3.5e-07
int colPeak = rep.addColumn("% peak", 9, 1);               // 88.6
rep.setCell(v, colErr, maxerr);
rep.setCell(v, colPeak, 100.0 * gbs / PEAK_GBS);
```

`setCell` also takes a string for a non-numeric cell, and an unset cell prints `-`.
Headers become CSV field names (`% peak` -> `pct_peak`).

A benchmark that measures something other than wall time -- `memory-access/chase`
reports cycles per load from `clock64()` -- calls `setShowTimes(false)` and lives
entirely in these columns.

## gemm_report.cuh

The GEMM report: one row per timed variant, so a file that races its own kernel
against cuBLAS just registers two.

```cpp
GemmReport rep("tiled_gemm", M, N, K, iter);
rep.setLaunch(blocksPerGrid, threadsPerBlock);
int tiled = rep.addVariant("tiled");

for (int i = 0; i <= iter; i++) {
    ...
    if (i == 0)  // warm-up: verified, then thrown away
        rep.setCheck(tiled, gemmCheckCpu(M, N, K, h_a, h_b, h_c, 1024));
    else
        rep.addSample(tiled, kernelTime);
}

rep.print(csv);
return rep.passed() ? 0 : 1;
```

`gemmCheckCpu(..., limit)` recomputes the top-left `limit x limit` corner of C on
the CPU and returns the mismatch count; the corner is capped because the
O(M*N*K) reference is far slower than the kernel under test. Half-precision
inputs report their footprint correctly after `setElementBytes(2, 4)`.

`print(false)` writes the human table; `print(true)` writes a CSV header plus one
row per variant, for sweeping from a script:

```
name,variant,m,n,k,iter,avg_ms,min_ms,max_ms,gflops,status
tiled_gemm,tiled,2000,2100,1900,10,6.6075,6.6028,6.6362,2415.42,pass
```
