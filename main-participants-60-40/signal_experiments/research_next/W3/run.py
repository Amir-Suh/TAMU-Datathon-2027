"""Compare ridge residuals using sparse and all-pair signed lag features."""
import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
SUITE = HERE.parent
EXPERIMENTS = SUITE.parent
KIT = EXPERIMENTS.parent
sys.path[:0] = [str(SUITE / "C1"), str(KIT), str(EXPERIMENTS)]
from run import BASE, frozen_models, joint, positive_mix
from models.common.crossfit import subset_split
from models.common.data import load_split, truncate
from models.m3_lag_pair_flip.model import FlipLagPairModel
from score import compare, sha256
from slot_forecasting.io import read_submission, write_submission
from synthetic import generate, mask_history

SHIFTS = (0, 4, 8, 16, 24, 32)
ARMS = ("main_three", "train_three", "all_28", "train_three_full_categories")
SIGMA = .25


def load_folds(train):
    path = EXPERIMENTS / "folds.csv"
    with path.open(newline="") as file:
        groups = {r["episode_id"]: int(r["fold"]) for r in csv.DictReader(file) if r["split"] == "train"}
    if set(groups) != set(train.episode_ids):
        raise ValueError("Training folds do not match the data.")
    return np.array([groups[episode] for episode in train.episode_ids])


def features(model, split, pairs):
    data = replace(split, labels=None, targets=None, reference=None, conditions=None)
    result = model._infer(data)
    ps = result["ind"]["post"].sum(axis=3)
    del result["core"]
    grid = np.asarray(model.pairs)
    rows = np.arange(split.n)[:, None]
    a, b = (split.cutoff[:, None, None] - grid[None, :, :]).transpose(2, 0, 1)
    first, second = split.X[rows, a], split.X[rows, b]
    known = (first >= 0) & (second >= 0)
    differ = (first > 0) != (second > 0)
    mass = ps.sum(axis=2)
    signed = np.where(differ, ps[:, :, 1] - ps[:, :, 0], ps[:, :, 0] - ps[:, :, 1]) * known
    common = np.column_stack((np.eye(3)[split.machine_type], mass.max(axis=1), result["ind"]["post0"]))
    codes = first * 5 + second
    canonical = np.array([model.pairs.index(p) for p in ((1, 3), (2, 5), (4, 7))])
    selected = np.array([model.pairs.index(p) for p in pairs])
    return {
        "main_three": np.column_stack((common, mass[:, canonical], signed[:, canonical])),
        "train_three": np.column_stack((common, mass[:, selected], signed[:, selected])),
        "all_28": np.column_stack((common, mass, signed)),
        "train_three_full_categories": np.column_stack(
            (common, mass[:, selected], signed[:, selected],
             ((codes[:, selected, None] == np.arange(25)) * known[:, selected, None] *
              mass[:, selected, None]).reshape(split.n, -1))),
    }, result["p_win"]


def crossfit(train, strength):
    folds = load_folds(train)
    rows_by_arm = {name: [] for name in ARMS}
    offset, labels, weights, records = [], [], [], {}
    for fold in sorted(set(folds)):
        fit = subset_split(train, np.flatnonzero(folds != fold))
        held = subset_split(train, np.flatnonzero(folds == fold))
        model = FlipLagPairModel()
        model.GRID = [{"kappa": strength}]
        model.fit(fit)
        selected = [model.pairs[i] for i in np.argsort(-model.pair_prior)[:3]]
        records[str(fold)] = {"fit_episodes": len(set(fit.episode_ids)), "held_episodes": len(set(held.episode_ids)),
                              "train_selected_pairs_in_order": selected}
        for shift in SHIFTS:
            sample = held if shift == 0 else replace(truncate(held, shift), signal=np.zeros(held.n, dtype=int))
            X, win = features(model, sample, selected)
            use = sample.labels >= 0
            for name in ARMS:
                rows_by_arm[name].append(X[name][use])
            offset.append(win[use])
            labels.append(sample.labels[use] > 0)
            weights.append(np.full(use.sum(), 1.0 if shift == 0 else .5))
    return ({k: np.concatenate(v) for k, v in rows_by_arm.items()},
            np.concatenate(offset), np.concatenate(labels), np.concatenate(weights), records)


def fit_offset(x, base, outcome, weight):
    location = x.mean(axis=0)
    scale = x.std(axis=0).clip(.10)
    design = np.column_stack((np.ones(len(x)), (x - location) / scale))
    offset = logit(np.clip(base, 1e-6, 1 - 1e-6))

    def loss(b):
        eta = offset + design @ b
        diff = weight * (expit(eta) - outcome)
        value = weight @ (np.logaddexp(0, eta) - outcome * eta) + np.square(b).sum() / (2 * SIGMA**2)
        return value, design.T @ diff + b / SIGMA**2

    result = minimize(loss, np.zeros(design.shape[1]), method="L-BFGS-B", jac=True,
                      options={"maxiter": 350, "ftol": 1e-12})
    if not result.success:
        raise ValueError(result.message)
    return {"location": location.tolist(), "scale": scale.tolist(), "coefficients": result.x.tolist()}


def apply(x, base, fit):
    design = np.column_stack((np.ones(len(x)), (x - fit["location"]) / fit["scale"]))
    return expit(logit(np.clip(base, 1e-6, 1 - 1e-6)) + design @ fit["coefficients"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    train = load_split(KIT / "public", "train")
    model = frozen_models()["m3"]
    X, base, labels, weights, records = crossfit(train, model.params["kappa"])
    print("Cross-fitted", len(base), "causal cutoff and pseudo-cutoff training rows.", flush=True)
    fits = {name: fit_offset(X[name], base, labels, weights) for name in ARMS}
    val = load_split(KIT / "public", "validation")
    by_id = read_submission(BASE, val.query_ids)
    raw = np.array([by_id[key] for key in val.query_ids])
    mix = positive_mix(raw)
    selected = [model.pairs[i] for i in np.argsort(-model.pair_prior)[:3]]
    V, m3_win = features(model, val, selected)
    small = subset_split(val, np.array([0, 3, 12]))
    order = np.array([2, 1, 0])
    a, _ = features(model, small, selected)
    b, _ = features(model, subset_split(small, order), selected)
    if any(not np.allclose(a[name][order], b[name], rtol=0, atol=1e-14) for name in ARMS):
        raise ValueError("Feature rows depend on input order.")
    paths = {"m3_same_mix": args.out / "m3_same_mix.csv"}
    write_submission(paths["m3_same_mix"], dict(zip(val.query_ids, joint(m3_win, mix))))
    for name in ARMS:
        for tag, win in (("", 1 - raw[:, 0]), ("_native", m3_win)):
            key = name + tag
            p = joint(apply(V[name], win, fits[name]), mix)
            paths[key] = args.out / f"{key}.csv"
            write_submission(paths[key], dict(zip(val.query_ids, p)))
    plan = {"train_rows": len(base), "shifts": SHIFTS, "penalty_sigma": SIGMA, "folds": records,
            "train_selected_full_data": selected, "base_forecast_hash": sha256(BASE),
            "source_sha256": sha256(__file__), "fits": fits,
            "historical_signal": "Neutral on shifts >0; true query signal at shift=0.",
            "scope": "Cutoff and pseudo-cutoff residual fit with episode-held-out M3; no main validation fitting. Hybrid-head transport scored separately from native M3 offsets."}
    (args.out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    report = compare(KIT / "public/validation", BASE, paths, 60)
    (args.out / "scores.json").write_text(json.dumps(report, indent=2) + "\n")
    law = generate("stable", episodes=12, seed=73663, rare=True, four_cells=True)
    synth = {}
    for name, data in (("full", law.split), ("sparse", mask_history(law.split, 24))):
        v, p = features(model, data, selected)
        scores = {"m3": float(np.mean((p - (1-data.targets[:, 0]))**2))}
        for arm in ARMS:
            scores[arm] = float(np.mean((apply(v[arm], p, fits[arm]) - (1-data.targets[:, 0]))**2))
        synth[name] = scores
    (args.out / "synthetic_scores.json").write_text(json.dumps(synth, indent=2) + "\n")
    print({k: v["points"] for k, v in report["models"].items()}, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
