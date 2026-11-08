#!/bin/sh
set -eu
d=signal_experiments/next_stage/20261107_dispatch/sparse_filter/attempt_01
audit=signal_experiments/next_stage/20261107_dispatch/coordinator/audit_run.py
archive=/private/tmp/fredy-replay.XxrBae
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$archive/deps" MPLCONFIGDIR="$d/smoke"
case "${1:-}" in
  smoke) python3 -B "$audit" --stage fit-predict --observe-root "$archive" --trace "$d/smoke/fit_archive_audit.jsonl" "$d/fit_predict.py" --out "$d/smoke" --smoke ;;
  fit) python3 -B "$audit" --stage fit-predict --observe-root "$archive" --trace "$d/fit_predict_audit.jsonl" "$d/fit_predict.py" ;;
  score) python3 -B "$audit" --stage score --trace "$d/score_audit.jsonl" "$d/score_frozen.py" ;;
esac
