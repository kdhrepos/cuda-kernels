#!/bin/bash
# Sum reduction, N = 2^26 floats.
# Prints CSV on stdout; tools/sweep.py collects it into results/. Edit the
# arguments here to change what gets measured.
set -euo pipefail
cd "$(dirname "$0")"

./reduction 67108864 --csv
