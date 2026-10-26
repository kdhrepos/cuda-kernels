#!/bin/bash
# Vector add with copies, n = 2^26 ints.
# Prints CSV on stdout; tools/sweep.py collects it into results/. Edit the
# arguments here to change what gets measured.
set -euo pipefail
cd "$(dirname "$0")"

./vecadd 67108864 --csv
