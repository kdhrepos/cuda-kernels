#!/bin/bash
# Matrix transpose, 4096 x 4096 floats.
# Prints CSV on stdout; tools/sweep.py collects it into results/. Edit the
# arguments here to change what gets measured.
set -euo pipefail
cd "$(dirname "$0")"

./transpose 4096 4096 --csv
