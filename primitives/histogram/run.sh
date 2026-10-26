#!/bin/bash
# 256-bin histogram, N = 2^28 bytes.
# Prints CSV on stdout; tools/sweep.py collects it into results/. Edit the
# arguments here to change what gets measured.
set -euo pipefail
cd "$(dirname "$0")"

./histogram 268435456 --csv
