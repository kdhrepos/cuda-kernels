#!/bin/bash
# WMMA tensor-core GEMM, FP16 in / FP32 out, 4096^3.
# Prints CSV on stdout; tools/sweep.py collects it into results/. Edit the
# arguments here to change what gets measured.
set -euo pipefail
cd "$(dirname "$0")"

./tensor_core_gemm_shared 4096 4096 4096 --csv
