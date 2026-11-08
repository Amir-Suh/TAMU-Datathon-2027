#!/usr/bin/env bash
# Run one Fredy dev arm from a fresh archive and retain its forecasts and fitted model.
set -euo pipefail

experiment=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
kit=$(cd "$experiment/../.." && pwd)
python=${PYTHON:-python3}
if (( $# < 1 || $# > 3 )); then
    echo "Usage: PYTHON=python3 bash replay.sh VARIANT [SEED] [cpu|mps]" >&2
    exit 2
fi
variant=$1
seed=${2:-1}
device=${3:-cpu}
case "$variant" in
    pair_l31|pair_trans_64_l15)
        arm=gbdt
        name=$variant
        args=(--set threads=2)
        ;;
    gru128_nomsg|gru128_factor)
        arm=seq
        name="${variant}_seed_$seed"
        args=(--set "seed=$seed" --set "device=$device")
        ;;
    *)
        echo "Unknown replay variant: $variant" >&2
        exit 2
        ;;
esac
if [[ "$device" != cpu && "$device" != mps ]]; then
    echo "Unknown training device: $device" >&2
    exit 2
fi

work=$(mktemp -d "${TMPDIR:-/tmp}/fredy-replay.XXXXXX")
trap 'rm -rf "$work"' EXIT
git -C "$kit/.." archive bbff7379e57984eb623588273ed1c958bb8b4f47 | tar -x -C "$work"
cd "$work"
if [[ "$arm" == seq && "$device" == mps ]]; then
    patch -p1 < "$experiment/fused_gru.patch"
fi
export PYTHONPATH="$work/src:$kit${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2

if [[ "$arm" == seq ]]; then
    runner=(-c 'import torch; torch.set_num_threads(2); torch.set_num_interop_threads(1); from tdx.run import main; main()')
else
    runner=(-m tdx.run)
fi
"$python" "${runner[@]}" --arm "$arm" --variant "$variant" --mode dev \
    --data "$kit/public" --out "$work/result" "${args[@]}"

mkdir -p "$experiment/forecasts" "$experiment/test_forecasts" "$experiment/checkpoints/$name"
cp "$work/result/val_pred.csv" "$experiment/forecasts/$name.csv"
cp "$work/result/test_pred.csv" "$experiment/test_forecasts/$name.csv"
cp "$work/result/meta.json" "$experiment/checkpoints/$name/run.json"
cp "$work/result/score.json" "$experiment/checkpoints/$name/fredy_score.json"
if [[ "$arm" == gbdt ]]; then
    cp "$work/result/model/features.json" "$experiment/checkpoints/$name/"
    gzip -n -c "$work/result/model/model.txt" > "$experiment/checkpoints/$name/model.txt.gz"
    mkdir -p "$experiment/train_forecasts"
    "$python" - "$kit/public" "$work/result/model" "$experiment/train_forecasts/$name.csv" <<'PY'
import sys
from slot_forecasting.io import write_submission
from tdx.arms.gbdt.arm import GBDTArm
from tdx.data import load_round
from tdx.score import prediction_map

data, model, output = sys.argv[1:]
train = load_round(data, "main", answers=False).splits["train"]
predictions = GBDTArm.load(model).predict(train)
write_submission(output, prediction_map(train, predictions))
PY
else
    cp "$work/result/model/"* "$experiment/checkpoints/$name/"
fi
