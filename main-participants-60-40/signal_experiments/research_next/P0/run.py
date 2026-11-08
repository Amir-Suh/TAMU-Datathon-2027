"""Estimate type-specific positive payout tails from training histories."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parents[1]
KIT = EXPERIMENTS.parent
sys.path[:0] = [str(KIT), str(KIT / "m5_m6_repair"), str(EXPERIMENTS)]
from models.common.data import CONDITIONS, TYPES, load_split
from score import compare, sha256
from slot_forecasting.io import read_queries, read_submission, write_submission

BASE = EXPERIMENTS / "combined"
DEFAULT_VALIDATION = BASE / "m3_match_split_win_pair_l31_factor_gru_size.csv"
DEFAULT_TEST = BASE / "m3_match_split_win_pair_l31_factor_gru_size_test.csv"
VARIANTS = {"raw_counts": 0.0, "half_count": 0.5}


def counts(x, machine_type):
    return np.array([[np.count_nonzero(x[machine_type == i] == k) for k in (2, 3, 4)]
                     for i in range(len(TYPES))])


def fit(x, machine_type, alpha):
    n = counts(x, machine_type) + alpha
    if np.any(n.sum(axis=1) == 0):
        raise ValueError("At least one training payout beyond 1x is needed for every machine type.")
    return n / n.sum(axis=1, keepdims=True)


def predict(query_path, source, ratios, output):
    queries = read_queries(query_path)
    keys = [q.query_id for q in queries]
    by_id = read_submission(source, keys)
    forecasts = {}
    for q in queries:
        base = by_id[q.query_id]
        tail_mass = sum(base[2:])
        forecasts[q.query_id] = (*base[:2], *(tail_mass * ratios[TYPES.index(q.machine_type)]))
    write_submission(output, forecasts)
    checked = read_submission(output, keys)
    maximum = max(abs(checked[key][i] - by_id[key][i]) for key in keys for i in (0, 1))
    if maximum > 1e-12:
        raise ValueError("Projection changed p0 or p1.")
    return maximum


def cutoff_diagnostics(train, path):
    with path.open(newline="") as file:
        by_episode = {r["episode_id"]: int(r["fold"]) for r in csv.DictReader(file)
                      if r["split"] == "train"}
    if set(train.episode_ids) != set(by_episode):
        raise ValueError("Fold manifest does not cover this training split.")
    fold_id = np.asarray([by_episode[key] for key in train.episode_ids])
    eligible = train.labels >= 2
    result = {}
    for name, alpha in VARIANTS.items():
        p = np.empty(train.n)
        for fold in np.unique(fold_id):
            held = fold_id == fold
            ratios = fit(train.X[~held], train.machine_type[~held], alpha)
            use = held & eligible
            p[use] = ratios[train.machine_type[use], train.labels[use] - 2]
        result[name] = {
            "cutoff_tail_outcomes": int(eligible.sum()),
            "mean_logloss": float(-np.log(np.clip(p[eligible], 1e-12, 1)).mean()),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=KIT / "public")
    parser.add_argument("--validation-base", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--test-base", type=Path)
    parser.add_argument("--folds", type=Path, default=EXPERIMENTS / "folds.csv")
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    train = load_split(args.data, "train")
    forecasts = {}
    diagnostics = {}
    for name, alpha in VARIANTS.items():
        ratios = fit(train.X, train.machine_type, alpha)
        forecasts[name] = args.out / f"{name}.csv"
        diagnostics[name] = {
            "pseudocount_per_class": alpha,
            "ratios": dict(zip(TYPES, ratios.tolist())),
            "validation_max_p0_p1_error": predict(args.data / "validation/queries.csv",
                                                   args.validation_base, ratios, forecasts[name]),
        }
        if args.test_base:
            diagnostics[name]["test_max_p0_p1_error"] = predict(
                args.data / "test/queries.csv", args.test_base, ratios, args.out / f"{name}_test.csv")
    meta = json.loads((args.data / "metadata.json").read_text())
    report = compare(args.data / "validation", args.validation_base, forecasts, meta["max_points"])
    (args.out / "scores.json").write_text(json.dumps(report, indent=2) + "\n")
    fitted = {
        "fit": "Observed training history only; cutoff labels are used in a separate grouped diagnostic.",
        "metadata": meta,
        "inputs": {name: sha256(args.data / "train" / name) for name in ("queries.csv", "history.csv", "labels.csv")},
        "source_sha256": sha256(__file__),
        "variants": diagnostics,
        "counts_by_type": dict(zip(TYPES, counts(train.X, train.machine_type).tolist())),
        "counts_by_condition_and_type": {
            c: dict(zip(TYPES, counts(train.X[train.conditions == c],
                                       train.machine_type[train.conditions == c]).tolist()))
            for c in CONDITIONS
        },
        "training_cutoff_OOF": cutoff_diagnostics(train, args.folds) if args.folds.is_file() else None,
    }
    (args.out / "plan.json").write_text(json.dumps(fitted, indent=2) + "\n")
    for name, result in report["models"].items():
        print(f"{name:20s} {result['points']:.9f} {result['delta_R']:+.9f} {result['delta_R_95']}")


if __name__ == "__main__":
    main()
