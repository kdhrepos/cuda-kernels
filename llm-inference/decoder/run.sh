#!/bin/bash
# Decode GEMV and KV-cache attention, T = 4096, 16 splits.
# Prints CSV on stdout; tools/sweep.py collects it into results/. Edit the
# arguments here to change what gets measured.
set -euo pipefail
cd "$(dirname "$0")"

./decoder 4096 16 --csv
