"""Benchmark a pinned local Chronos sample forecast as a categorical distribution."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
from chronos import ChronosPipeline

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parents[1]
KIT = EXPERIMENTS.parent
sys.path[:0] = [str(KIT), str(KIT / "m5_m6_repair"), str(EXPERIMENTS)]
from models.common.data import MISSING, PAYOUTS, load_split
from score import compare, sha256
from slot_forecasting.io import read_submission, write_submission

BASE = EXPERIMENTS / "combined/m3_match_split_win_pair_l31_factor_gru_size.csv"
BINS = np.array([.5, 1.5, 3.5, 12.5])
SAMPLES = 128
SEED = 6711
BATCH = 16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    if not (args.checkpoint / "model.safetensors").is_file():
        raise ValueError("Expected a local safetensors snapshot.")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    pipeline = ChronosPipeline.from_pretrained(str(args.checkpoint), local_files_only=True,
                                               trust_remote_code=False)
    val = load_split(KIT / "public", "validation")
    start = perf_counter()
    y = np.where(val.X != MISSING, np.asarray(PAYOUTS)[val.X.clip(0)], np.nan)
    contexts = [torch.as_tensor(y[i, :c], dtype=torch.float32) for i, c in enumerate(val.cutoff)]
    forecast = np.empty((val.n, len(PAYOUTS)))
    with torch.inference_mode():
        for start_row in range(0, val.n, BATCH):
            inputs = contexts[start_row:start_row+BATCH]
            draws = pipeline.predict(inputs, prediction_length=1, num_samples=SAMPLES)
            if tuple(draws.shape) != (len(inputs), SAMPLES, 1):
                raise ValueError(f"Unexpected forecast shape {draws.shape}.")
            raw = draws[:, :, 0].numpy()
            if not np.isfinite(raw).all():
                raise ValueError("Nonfinite Chronos samples.")
            categories = np.digitize(raw, BINS)
            probabilities = np.stack([(categories == k).mean(axis=1) for k in range(len(PAYOUTS))], axis=1)
            forecast[start_row:start_row+len(inputs)] = probabilities
    blend = read_submission(BASE, val.query_ids)
    mixed = .90 * np.array([blend[q] for q in val.query_ids]) + .10 * forecast
    paths = {}
    for name, result in (("chronos_tiny", forecast), ("fixed_10_percent", mixed)):
        paths[name] = args.out / f"{name}.csv"
        write_submission(paths[name], dict(zip(val.query_ids, result)))
    report = compare(KIT / "public/validation", BASE, paths, 60)
    (args.out / "scores.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "plan.json").write_text(json.dumps({
        "checkpoint": "amazon/chronos-t5-tiny", "revision": args.checkpoint.name,
        "checkpoint_sha256": sha256(args.checkpoint / "model.safetensors"),
        "torch": torch.__version__, "sample_count": SAMPLES, "seed": SEED,
        "sample_bin_boundaries": BINS.tolist(), "batch": BATCH, "elapsed_seconds": perf_counter()-start,
        "inference": "Offline CPU; missing ticks passed as NaN; evaluation only.",
    }, indent=2) + "\n")
    print({name: model["points"] for name, model in report["models"].items()}, flush=True)


if __name__ == "__main__":
    main()
