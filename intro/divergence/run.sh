#!/bin/bash
# Warp divergence, n = 2^22, 5000 FMAs per thread, k-way switch for k = 1..32.
# A k column is added in front of every row, since the CSV does not carry k.
# Prints CSV on stdout; tools/sweep.py collects it into results/. Edit the
# arguments here to change what gets measured.
set -euo pipefail
cd "$(dirname "$0")"

for k in 1 2 4 8 16 32; do
    ./divergence $k 4194304 5000 --csv | sed "s/^name,/k,name,/; t; s/^/$k,/"
done
