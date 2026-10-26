#!/bin/bash
# Naive GEMM and its swapped mapping:
#   size  : 16x16 blocks, four sizes (naive only)
#   block : 4096^3, eleven block shapes (both binaries)
# bx,by columns are added in front of every row, since the CSV does not carry them.
# Prints CSV on stdout; tools/sweep.py collects it into results/. Edit the
# arguments here to change what gets measured.
set -euo pipefail
cd "$(dirname "$0")"

run() {  # run <binary> <bx> <by> <M> <N> <K>
    ./$1 $2 $3 $4 $5 $6 --csv | sed "s/^name,/bx,by,name,/; t; s/^/$2,$3,/"
}

for mnk in "1000 1100 900" "2000 2100 1900" "3000 3100 2900" "4000 4100 3900"; do
    run naive_gemm 16 16 $mnk
done

for shape in 1x256 2x128 4x64 8x32 16x16 32x8 64x4 128x2 256x1 8x8 32x32; do
    for bin in naive_gemm naive_gemm_swapped; do
        run $bin ${shape%x*} ${shape#*x} 4096 4096 4096
    done
done
