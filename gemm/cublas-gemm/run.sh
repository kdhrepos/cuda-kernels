#!/bin/bash
# cuBLAS SGEMM, 4096^3.
# Prints CSV on stdout; tools/sweep.py collects it into results/. Edit the
# arguments here to change what gets measured.
set -euo pipefail
cd "$(dirname "$0")"

./cublas_sgemm 4096 4096 4096 --csv
