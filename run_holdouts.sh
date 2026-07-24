#!/bin/bash
# Run the bucketing pipeline on one or more holdout benchmarks and print, for
# each, the two numbers the results sheet wants: Balanced Accuracy + runtime.
#
# Usage:
#   ./run_holdouts.sh holdout                 # one benchmark
#   ./run_holdouts.sh holdout1 holdout2 ...   # several
#
# Each argument is a directory containing:
#   input.csv   (Case, Regr Log, Sim Log, Trace Log)
#   golden.csv  (Case, Bug)          <- used only for scoring + to derive K
#
# K (bucket count) is taken from the number of distinct bugs in golden.csv,
# i.e. the same K the grader would pass via --k.

PY=.venv/bin/python

printf "%-22s %-10s %-10s %s\n" "holdout" "accuracy" "runtime" "K"
printf "%-22s %-10s %-10s %s\n" "----------------------" "--------" "--------" "---"

for d in "$@"; do
  name=$(basename "$d")
  in="$d/input.csv"; gold="$d/golden.csv"; out="$d/output.csv"

  if [ ! -f "$in" ] || [ ! -f "$gold" ]; then
    printf "%-22s %s\n" "$name" "MISSING input.csv or golden.csv"
    continue
  fi

  # K = number of distinct bug labels in golden.csv (skip header)
  k=$(tail -n +2 "$gold" | cut -d, -f2 | sort -u | grep -c .)

  start=$($PY -c 'import time;print(time.time())')
  $PY regr_fail_bucketing --input "$in" --output "$out" --k "$k" >/dev/null 2>&1
  end=$($PY -c 'import time;print(time.time())')

  rt=$($PY -c "print(f'{$end-$start:.2f}s')")
  ba=$($PY evaluate.py --pred "$out" --gold "$gold" | grep -oE '[0-9.]+$')

  printf "%-22s %-10s %-10s %s\n" "$name" "$ba" "$rt" "$k"
done
