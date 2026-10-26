#!/bin/bash
# Strided read and write, N = 2^24 and 2^26 floats (strides 1..1024 inside).
# An N column is added in front of every row, since the CSV does not carry N.
# Prints CSV on stdout; tools/sweep.py collects it into results/. Edit the
# arguments here to change what gets measured.
set -euo pipefail
cd "$(dirname "$0")"

for n in 16777216 67108864; do
    ./stride $n --csv | sed "s/^name,/N,name,/; t; s/^/$n,/"
done
