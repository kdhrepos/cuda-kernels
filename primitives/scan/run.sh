#!/bin/bash
# Inclusive scan, N = 2^28 ints.
# Prints CSV on stdout; tools/sweep.py collects it into results/. Edit the
# arguments here to change what gets measured.
set -euo pipefail
cd "$(dirname "$0")"

./scan 268435456 --csv
