# normalization

Two row-wise kernels with one 256-thread block per row; sizes are fixed in the source.
softmax turns a `float` [2048 × 32768] into row probabilities, and RMSNorm turns a
`float` [16384 × 4096] into x ÷ rms(x) × gamma. Each comes in unfused and fused versions.

| Variant | How |
| --- | --- |
| softmax unfused | separate max, sum and normalize kernels |
| softmax fused3 | one kernel, still three passes over x |
| softmax online | running max and sum in one pass, then normalize |
| rmsnorm unfused | separate sum-of-squares and normalize kernels |
| rmsnorm fused | one kernel, each thread keeps its 16 values in registers |

## Run

```bash
make ARCH=sm_86
./softmax                   # no arguments; same for ./rms_norm
./softmax --csv
./run.sh                    # reproduces the Result table
```

Every element is checked against a CPU reference accumulated in double.

## Result

RTX 3090, kernel time only, 1 warm-up + 10 runs.

GB/s = one read of x + one write of y ÷ kernel time (useful bytes).

| Variant | GB/s | × unfused | Note |
| --- | ---: | ---: | --- |
| softmax unfused | 412 | 1.00 | every variant runs at ~825–855 GB/s of actual traffic, so the speedup is the ratio of bytes moved |
| softmax fused3 | 425 | 1.03 | same traffic as unfused (x read 3 times + y written), only the kernel boundaries go away |
| softmax online | 563 | 1.37 | running max and sum in one pass, x read twice → 3 passes over N instead of 4 |
| rmsnorm unfused | 568 | 1.00 | |
| rmsnorm fused | 841 | 1.48 | each thread keeps its 16 values in registers, x read once → 2N instead of 3N |
