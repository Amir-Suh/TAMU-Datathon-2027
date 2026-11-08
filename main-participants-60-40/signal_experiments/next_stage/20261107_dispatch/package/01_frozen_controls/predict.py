"""Predict House Edge main queries from read-only frozen weights and histories."""
import argparse
import gzip
import json
from pathlib import Path
import os

# Set numerical thread limits before importing libraries.
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np

from input_only import input_only
from models.m3_lag_pair_flip.model import FlipLagPairModel
from slot_forecasting.io import require_ids, write_submission

WEIGHTS = Path(__file__).resolve().parent / "weights"


def load_m3():
    state = json.loads((WEIGHTS / "m3.json").read_text())
    size = state.pop("size")
    model = FlipLagPairModel()
    for name in ("tau_grid", "win_prior", "hi", "lo", "pair_prior", "tau_q"):
        state[name] = np.asarray(state[name])
    vars(model).update(state)
    half_life = float("inf") if size["half_life"] is None else size["half_life"]
    model.size.params = {"strength": size["strength"], "half_life": half_life}
    model.size.prior = np.asarray(size["prior"])
    return model


def predict_hybrid(model, flat_split, root, split_name):
    import lightgbm as lgb
    from match_split import predict
    from tdx.arms.gbdt.arm import GBDTArm
    from tdx.arms.seq.arm import SeqArm
    from tdx.data import _read_split

    prior = json.loads((WEIGHTS / "match_contrast.json").read_text())
    win = predict(model, flat_split, np.asarray(prior["learned_contrast_11_minus_00"]))
    p0 = win["candidate"][:, :1]

    split = _read_split(root, split_name)
    tree_dir = WEIGHTS / "pair_l31"
    meta = json.loads((tree_dir / "features.json").read_text())
    meta["params"]["threads"] = 1
    tree = GBDTArm(**meta["params"])
    tree.names, tree.keep = meta["names"], np.asarray(meta["keep"], dtype=bool)
    with gzip.open(tree_dir / "model.txt.gz", "rt", encoding="utf-8") as source:
        tree.model = lgb.Booster(model_str=source.read())
    tree_prob = tree.predict(split)
    seeds = [SeqArm.load(WEIGHTS / f"gru128_factor_seed_{seed}").predict(split)
             for seed in range(1, 4)]
    donor = (tree_prob + np.mean(seeds, axis=0)) / 2
    by_id = dict(zip(split.query_ids.ravel(), donor.reshape(-1, 5)))
    require_ids(by_id, flat_split.query_ids)
    size = np.array([by_id[key][1:] for key in flat_split.query_ids])
    return np.concatenate([p0, (1 - p0) * size / size.sum(axis=1, keepdims=True)], axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("m3", "hybrid"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    args = parser.parse_args()
    model = load_m3()
    split = input_only(args.data, args.split)
    if args.model == "hybrid":
        probabilities = predict_hybrid(model, split, args.data, args.split)
    else:
        probabilities = model.predict(split)
    if len(set(split.query_ids)) != len(split.query_ids):
        raise ValueError("Duplicate query IDs")
    write_submission(args.out, dict(zip(split.query_ids, probabilities)))


if __name__ == "__main__":
    main()
