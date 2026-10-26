#!/bin/bash
# 100 tiny kernels, normal launch vs CUDA Graph.
# Prints CSV on stdout; tools/sweep.py collects it into results/. Edit the
# arguments here to change what gets measured.
set -euo pipefail
cd "$(dirname "$0")"

./graph_launch --csv
