#!/bin/bash
# Matrix add, 9999 x 6001 (fixed in source), over block shapes bx x by: the nine
# 256-thread shapes plus 8x8 and 32x32.
# bx,by columns are added in front of every row, since the CSV does not carry them.
# Prints CSV on stdout; tools/sweep.py collects it into results/. Edit the
# arguments here to change what gets measured.
set -euo pipefail
cd "$(dirname "$0")"

for shape in 1x256 2x128 4x64 8x32 16x16 32x8 64x4 128x2 256x1 8x8 32x32; do
    bx=${shape%x*} by=${shape#*x}
    ./matadd $bx $by --csv | sed "s/^name,/bx,by,name,/; t; s/^/$bx,$by,/"
done
