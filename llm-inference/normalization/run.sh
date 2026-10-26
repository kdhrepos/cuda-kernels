#!/bin/bash
# Softmax and RMSNorm (sizes fixed in source).
# Prints CSV on stdout; tools/sweep.py collects it into results/. Edit the
# arguments here to change what gets measured.
set -euo pipefail
cd "$(dirname "$0")"

./softmax --csv
./rms_norm --csv
