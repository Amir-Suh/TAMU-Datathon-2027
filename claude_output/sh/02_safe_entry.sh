#!/usr/bin/env bash
# Safe fallback entry: the kit's pooled_recent baseline (validation 30.826 points). Stdlib only, ~10 s.
# Produces outputs/safe/submission.csv and outputs/safe_entry.zip. Upload BOTH to the Main form now, replace later.
# Run:  ! bash claude_output/sh/02_safe_entry.sh
set -euo pipefail
cd "$(dirname "$0")/../../kit"
[ -e public ] || { echo "run 01_link_data.sh first"; exit 1; }
OUT=../outputs/safe
rm -rf "$OUT" ../outputs/safe_entry.zip
python3 run_starter.py --method pooled_recent --out "$OUT"
python3 check_submission.py "$OUT/submission.csv"
python3 check_submission.py "$OUT/validation_predictions.csv" --split validation
python3 build_entry.py --starter "$OUT" --out ../outputs/safe_entry.zip
sha256sum "$OUT/submission.csv" ../outputs/safe_entry.zip
echo "UPLOAD: $(cd .. && pwd)/outputs/safe/submission.csv  and  $(cd .. && pwd)/outputs/safe_entry.zip"
