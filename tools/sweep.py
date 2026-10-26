#!/usr/bin/env python3
"""Run benchmark run.sh scripts and save their CSV output under results/.

Every benchmark folder <group>/<kernel>/ has a run.sh that runs its binaries
with --csv and the arguments it wants; this only picks which ones to run and
stores what they print as results/<group>/<kernel>/<kernel>.csv ('-' -> '_').

    tools/sweep.py                       # every run.sh
    tools/sweep.py gemm                  # one group
    tools/sweep.py gemm/02-tiled primitives/scan

Build first (make ARCH=sm_86). A run may print several CSV tables; a new one
starts at every line with a "name" column, and all rows go into one file with
the union of the columns.
"""
import csv
import glob
import io
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_tables(text):
    rows, fields, header = [], [], None
    for rec in csv.reader(io.StringIO(text)):
        if not rec:
            continue
        if "name" in rec:
            header = rec
            fields += [f for f in rec if f not in fields]
        elif header:
            rows.append(dict(zip(header, rec)))
    return fields, rows


def main():
    if any(a.startswith("-") for a in sys.argv[1:]):
        print(__doc__)
        return 0
    targets = [t.strip("/") for t in sys.argv[1:]]
    scripts = sorted(glob.glob(os.path.join(ROOT, "*", "*", "run.sh")))
    bench = [os.path.relpath(os.path.dirname(s), ROOT) for s in scripts]
    if targets:
        bench = [b for b in bench if any(b == t or b.startswith(t + "/") for t in targets)]
    if not bench:
        print("no run.sh matches %s" % " ".join(targets), file=sys.stderr)
        return 1

    failed = []
    for b in bench:
        print("running %s ..." % b, file=sys.stderr, flush=True)
        p = subprocess.run([os.path.join(ROOT, b, "run.sh")], capture_output=True, text=True)
        fields, rows = parse_tables(p.stdout)
        if p.returncode != 0 or not rows:
            why = (p.stderr or p.stdout).strip().split("\n")[-1]
            print("  FAILED: %s" % (why or "no CSV output"), file=sys.stderr)
            failed.append(b)
            if not rows:
                continue
        out_dir = os.path.join(ROOT, "results", b)
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, os.path.basename(b).replace("-", "_") + ".csv")
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        nfail = sum(r.get("status") == "fail" for r in rows)
        if nfail:
            failed.append(b)
        print("  %d rows -> %s%s" % (len(rows), os.path.relpath(out, ROOT),
                                     "  (%d status=fail)" % nfail if nfail else ""),
              file=sys.stderr)

    if failed:
        print("\nfailed: %s" % " ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
