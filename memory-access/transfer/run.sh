#!/bin/bash
# Host<->device transfer methods, vector add on N = 2^26 ints.
# Prints CSV on stdout; tools/sweep.py collects it into results/. Edit the
# arguments here to change what gets measured.
set -euo pipefail
cd "$(dirname "$0")"

./transfer 26 --csv
