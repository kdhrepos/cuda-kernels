#!/bin/bash
# FP16 vs weight-only INT8 GEMV, 4096 x 4096.
# Prints CSV on stdout; tools/sweep.py collects it into results/. Edit the
# arguments here to change what gets measured.
set -euo pipefail
cd "$(dirname "$0")"

./w8_gemv --csv
