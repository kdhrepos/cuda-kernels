#!/bin/bash
# Pointer chase, 4 KB .. 64 MB arrays (inside), cycles per load.
# Prints CSV on stdout; tools/sweep.py collects it into results/. Edit the
# arguments here to change what gets measured.
set -euo pipefail
cd "$(dirname "$0")"

./p_chase --csv
