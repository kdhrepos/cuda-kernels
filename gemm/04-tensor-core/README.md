# 04-tensor-core

Matrix multiply on tensor cores: FP16 A×B → FP32 C, 4096³, with WMMA 16×16×16
fragments. 128-thread blocks (4 warps) each compute a 32×32 tile of C through one
shared-memory stage. The comparison point is cuBLAS with the same types.

| Variant | How |
| --- | --- |
| wmma | 4 warps per block, 32×32 C tile, one shared-memory stage |
| cublas_hgemm | cuBLAS `cublasGemmEx`, FP16 in, FP32 accumulate |

## Run

```bash
make ARCH=sm_86
./tensor_core_gemm_shared 4096 4096 4096          # M N K
./tensor_core_gemm_shared 4096 4096 4096 --csv
./run.sh                                          # reproduces the Result table
```

The top-left 1024×1024 corner of C is checked against the CPU.

## Result

RTX 3090, 4096³, kernel time only, 1 warm-up + 10 runs.
TFLOPS = 2×M×N×K ÷ kernel time.

| Variant | TFLOPS | vs cublas_hgemm | Why |
| --- | ---: | ---: | --- |
| wmma | 23.0 | | tensor cores double the compute ceiling, so the small 32×32 tile becomes the limit; DRAM-bound (Nsight: DRAM 74.8 %) |
| cublas_hgemm | | 1.00 | 256×128 tiles and a 3-stage copy pipeline, so each byte from DRAM is reused far more |

wmma reaches 32 % of the 71 TFLOPS FP16 tensor-core peak.
