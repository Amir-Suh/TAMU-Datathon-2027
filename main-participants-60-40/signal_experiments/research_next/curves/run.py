"""Measure a nested training-episode curve at fixed main validation cutoffs."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "C1"))
from run import BASE, KIT, frozen_models, joint, load_split, positive_mix, read_submission, write_submission
from models.common.crossfit import subset_split
from models.common.data import CONDITIONS
from models.m3_lag_pair_flip.model import FlipLagPairModel
from score import compare, sha256


def main():
    train = load_split(KIT / "public", "train")
    validation = load_split(KIT / "public", "validation")
    by_id = read_submission(BASE, validation.query_ids)
    mix = positive_mix(np.array([by_id[k] for k in validation.query_ids]))
    folds = {
        condition: sorted(set(train.episode_ids[train.conditions == condition]),
                          key=lambda ep: hashlib.sha256(f"train-curve-v1:{ep}".encode()).digest())
        for condition in CONDITIONS
    }
    frozen = frozen_models()["m3"]
    paths, fits = {}, {}
    for n in (24, 48, 96):
        episodes = {ep for condition in CONDITIONS for ep in folds[condition][:n // 4]}
        fit = subset_split(train, np.flatnonzero(np.isin(train.episode_ids, sorted(episodes))))
        model = FlipLagPairModel()
        model.GRID = [{"kappa": frozen.params["kappa"]}]
        model.fit(fit)
        pred = model.predict_win(validation)
        if n == 96 and not np.allclose(pred, frozen.predict_win(validation), rtol=0, atol=1e-12):
            raise ValueError("The 96-episode refit disagrees with frozen M3.")
        name = f"train_{n}"
        paths[name] = HERE / f"{name}.csv"
        write_submission(paths[name], dict(zip(validation.query_ids, joint(pred, mix))))
        fits[name] = {"episodes": sorted(episodes), "kappa": model.params["kappa"]}
    report = compare(KIT / "public/validation", BASE, paths, 60)
    (HERE / "scores.json").write_text(json.dumps(report, indent=2) + "\n")
    (HERE / "plan.json").write_text(json.dumps({"fits": fits, "source_sha256": sha256(__file__),
        "scope": "M3 win component only; frozen hybrid payout mix for each training budget."}, indent=2) + "\n")
    print({k: (v["points"], v["p0_mse"]) for k, v in report["models"].items()}, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
