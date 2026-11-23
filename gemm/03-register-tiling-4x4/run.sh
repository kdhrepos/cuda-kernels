#!/bin/bash
# Register-tiled GEMMs, 4096^3.
# Prints CSV on stdout; tools/sweep.py collects it into results/. Edit the
# arguments here to change what gets measured.
set -euo pipefail
cd "$(dirname "$0")"

for bin in thread_tile_1x4_strided thread_tile_1x4_contiguous thread_tile_4x4 \
           thread_tile_4x4_transA thread_tile_4x4_transA_pad; do
    ./$bin 4096 4096 4096 --csv
done
