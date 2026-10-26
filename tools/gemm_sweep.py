#!/usr/bin/env python3
"""Run a set of benchmarks over a set of sizes and collect the results.

Every benchmark here takes --csv and prints one row per variant, so a sweep is
just running them and concatenating. This adds the two things that are tedious
by hand: substituting the size into each benchmark's own argument form, and
pivoting the result into a table you can read.

    # square GEMMs at three sizes
    tools/sweep.py -s 1024,2048,4096 -a '{s} {s} {s}' gemm/*/[a-z]*[!.]

    # a benchmark whose CLI wants a block shape first
    tools/sweep.py -s 1024,2048 -a '16 16 {s} {s} {s}' gemm/01-naive/naive_gemm

    # one-argument benchmarks
    tools/sweep.py -s 1048576,4194304 -a '{s}' primitives/*/[a-z]*[!.]

    # keep the raw rows as well as the table
    tools/sweep.py -s 1024,2048 -a '{s} {s} {s}' --out sweep.csv gemm/02-tiled/tiled_gemm

Sizes that make a benchmark fail (out of memory, an unsupported shape) are
reported and skipped rather than killing the sweep.
"""
import argparse
import csv
import io
import os
import subprocess
import sys


def run_one(binary, args, timeout):
    cmd = [binary if os.path.isabs(binary) else "./" + binary] + args
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "timed out after %gs" % timeout
    rows = list(csv.DictReader(io.StringIO(p.stdout)))
    if not rows:
        why = (p.stderr or p.stdout).strip().split("\n")
        return None, (why[-1] if why else "no CSV output")
    return rows, None


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("binaries", nargs="+", help="benchmark executables to run")
    ap.add_argument("-s", "--sizes", required=True,
                    help="comma-separated sizes, e.g. 1024,2048,4096")
    ap.add_argument("-a", "--args", default="{s}",
                    help="argument template; {s} is replaced by the size "
                         "(default: '{s}')")
    ap.add_argument("-m", "--metric", default="gflops",
                    help="CSV column to pivot into the table "
                         "(default: gflops; try gbs, avg_ms)")
    ap.add_argument("--out", help="also write every raw CSV row here")
    ap.add_argument("--timeout", type=float, default=600.0,
                    help="seconds allowed per run (default: 600)")
    opt = ap.parse_args()

    sizes = [t.strip() for t in opt.sizes.split(",") if t.strip()]
    collected = []          # (label, size, row)
    failures = []

    for binary in opt.binaries:
        if os.path.isfile(binary) and not os.access(binary, os.X_OK):
            continue        # a glob also picked up a source file or Makefile
        if not os.path.isfile(binary):
            failures.append((binary, "-", "not built"))
            continue
        for size in sizes:
            args = opt.args.format(s=size).split() + ["--csv"]
            rows, err = run_one(binary, args, opt.timeout)
            if err:
                failures.append((binary, size, err))
                continue
            for r in rows:
                label = "%s/%s" % (r.get("name", os.path.basename(binary)),
                                   r.get("variant", "-"))
                collected.append((label, size, r))
            print("  ran %-46s %s" % (os.path.basename(binary), size),
                  file=sys.stderr)

    if not collected:
        print("nothing ran", file=sys.stderr)
        return 1

    if opt.out:
        fields = []
        for _, _, r in collected:
            for k in r:
                if k not in fields: fields.append(k)
        with open(opt.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["size"] + fields)
            w.writeheader()
            for _, size, r in collected:
                row = dict(r); row["size"] = size
                w.writerow(row)
        print("raw rows -> %s" % opt.out, file=sys.stderr)

    # pivot: one row per benchmark variant, one column per size
    labels, seen = [], set()
    for label, _, _ in collected:
        if label not in seen:
            seen.add(label); labels.append(label)
    cell = {(l, s): r.get(opt.metric, "") for l, s, r in collected}
    status = {(l, s): r.get("status", "") for l, s, r in collected}

    width = max([len(l) for l in labels] + [len("benchmark / variant")])
    print()
    print("%s = %s   (x marks a row whose check did not pass)"
          % (opt.metric, "higher is better" if opt.metric != "avg_ms" else "lower is better"))
    print()
    header = " %-*s" % (width, "benchmark / variant") + "".join("%14s" % s for s in sizes)
    print(header)
    print(" " + "-" * (len(header) - 1))
    for l in labels:
        line = " %-*s" % (width, l)
        for s in sizes:
            v = cell.get((l, s), "")
            if v == "":
                line += "%14s" % "-"
            else:
                mark = "" if status.get((l, s), "pass") in ("pass", "", "unchecked") else "x"
                line += "%14s" % (("%.1f" % float(v)) + mark)
        print(line)
    print(" " + "-" * (len(header) - 1))

    if failures:
        print("\nskipped:")
        for b, s, why in failures:
            print("  %-46s %-10s %s" % (os.path.basename(b), s, why))
    return 0


if __name__ == "__main__":
    sys.exit(main())
