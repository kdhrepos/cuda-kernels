# quantized-gemv

y = W·x with W stored as INT8 plus one FP32 scale per row (weight-only
quantization), dequantized inside the GEMV. W [4096 × 4096], x FP16, y FP32.
Four copies of W are read in turn so L2 cannot serve repeat reads.

| Variant | How |
| --- | --- |
| fp16 | FP16 weights, the baseline |
| int8 | INT8 weights × per-row FP32 scale, dequantized in registers |

## Run

```bash
make ARCH=sm_86
./w8_gemv                   # no arguments
./w8_gemv --csv
./run.sh                    # reproduces the Result table
```

Checked against a CPU reference. INT8 against the dequantized weights.

## Result

RTX 3090, 4096 × 4096, kernel time only, 1 warm-up + 10 runs.

GB/s = bytes the kernel must read ÷ kernel time.

| Variant | ms | GB/s | vs fp16 | Notes |
| --- | ---: | ---: | ---: | --- |
| fp16 | 0.0421 | 798 | 1.00 | |
| int8 | 0.0225 | 747 | 1.87 | half the bytes of W at about the same bandwidth, so close to the ×2.00 byte ratio |

Accuracy cost of INT8: compared with the output computed from the original FP16 weights,
the INT8 output vector is off by 0.85 % of its own size, i.e.
‖y_int8 − y_fp16‖ ÷ ‖y_fp16‖ = 0.85 % (vector length = square root of the sum of squares).
