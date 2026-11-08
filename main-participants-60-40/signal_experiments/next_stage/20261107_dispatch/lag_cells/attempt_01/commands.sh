#!/bin/sh
set -eu
d=signal_experiments/next_stage/20261107_dispatch
a="$d/lag_cells/attempt_01"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# The coordinator wrapper requires new trace filenames on replay.
tag="${1:-reproduce}"
python3 -B "$d/coordinator/audit_run.py" --stage fit-predict --trace "$a/replay_${tag}.jsonl" "$a/replay.py" > "$a/replay_${tag}.log" 2>&1
python3 -B "$d/coordinator/audit_run.py" --stage fit-predict --trace "$a/four_cells_${tag}.jsonl" "$a/four_cells.py" > "$a/four_cells_${tag}.log" 2>&1
python3 -B "$d/coordinator/audit_run.py" --stage score --trace "$a/score_${tag}.jsonl" "$a/score_frozen.py" > "$a/score_${tag}.log" 2>&1
for arm in W1a W1b W2 E3Frozen E3Full E3Shared; do
  python3 -B check_submission.py "$a/validation_$arm.csv" --split validation
  python3 -B check_submission.py "$a/test_$arm.csv"
done > "$a/format_checks_${tag}.log"
