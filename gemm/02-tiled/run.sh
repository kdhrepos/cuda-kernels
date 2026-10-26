#!/bin/bash
# Shared-memory tiled GEMM, 4096^3, for TILE = 8, 16, 32. TILE is a compile-time
# value, so each one is a rebuild; the default TILE 16 binary is rebuilt at the end.
# A TILE column is added in front of every row, since the CSV does not carry it.
# Prints CSV on stdout; tools/sweep.py collects it into results/. Edit the
# arguments here to change what gets measured.
set -euo pipefail
cd "$(dirname "$0")"
ARCH=${ARCH:-sm_86}

build() { make -s clean >&2 && make -s ARCH=$ARCH "$@" >&2; }
trap 'build' EXIT

for tile in 8 16 32; do
    build DEFINES="TILE=$tile"
    ./tiled_gemm 4096 4096 4096 --csv | sed "s/^name,/TILE,name,/; t; s/^/$tile,/"
done
