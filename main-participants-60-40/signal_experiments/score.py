"""Rescore standard forecast CSVs and compare their fixed episode losses."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
from slot_forecasting.contest import contest_score
from slot_forecasting.io import read_queries, read_submission, read_targets, require_ids
from slot_forecasting.schema import CONDITIONS

BASELINE = KIT / "m5_m6_repair/results/repaired-m5-m6/m3/validation_predictions.csv"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_folds(public_dir, out):
    """Assign four folds per split before training, balanced by episode condition."""
    with Path(out).open("w", newline="") as out_file:
        writer = csv.writer(out_file)
        writer.writerow(("split", "episode_id", "fold"))
        for split in ("train", "validation"):
            directory = Path(public_dir) / split
            queries = read_queries(directory / "queries.csv")
            with (directory / "conditions.csv").open(newline="") as file:
                conditions = {r["query_id"]: r["condition"] for r in csv.DictReader(file)}
            require_ids(conditions, {q.query_id for q in queries})
            episodes = defaultdict(set)
            for query in queries:
                episodes[query.episode_id].add(conditions[query.query_id])
            for condition in CONDITIONS:
                members = [key for key, labels in episodes.items() if labels == {condition}]
                members.sort(key=lambda key: hashlib.sha256(f"signal-v1:{key}".encode()).hexdigest())
                if not members or any(len(labels) != 1 for labels in episodes.values()):
                    raise ValueError("Missing condition or inconsistent episode grouping.")
                writer.writerows((split, key, idx % 4) for idx, key in enumerate(members))


def compare(data, baseline, forecasts, points, draws=10000):
    data = Path(data)
    queries = {q.query_id: q for q in read_queries(data / "queries.csv")}
    targets = read_targets(data / "targets.csv")
    require_ids(queries, {t.query_id for t in targets})
    ids = [t.query_id for t in targets]
    if any(queries[t.query_id].episode_id != t.episode_id for t in targets):
        raise ValueError("Target episodes disagree with the query manifest.")
    with (data / "conditions.csv").open(newline="") as file:
        groups = {r["query_id"]: r["condition"] for r in csv.DictReader(file)}
    reference = read_submission(data / "reference.csv", queries)
    truth = np.asarray([t.probabilities for t in targets])
    _, row_to_episode, episode_counts = np.unique(
        [t.episode_id for t in targets], return_inverse=True, return_counts=True
    )
    episode_condition = np.empty(len(episode_counts), dtype=object)
    for i, key in enumerate(ids):
        episode_condition[row_to_episode[i]] = groups[key]

    def episode_error(predictions):
        row_error = np.square(predictions - truth).sum(axis=1)
        return np.bincount(row_to_episode, weights=row_error) / episode_counts

    def array(predictions):
        return np.asarray([predictions[key] for key in ids])

    def conditional_mix(p):
        win_mass = p[:, 1:].sum(axis=1, keepdims=True)
        return np.divide(p[:, 1:], win_mass, out=np.zeros_like(p[:, 1:]), where=win_mass > 0)

    reference_error = episode_error(array(reference))
    rng = np.random.default_rng(20261107)
    strata = {c: np.flatnonzero(episode_condition == c) for c in CONDITIONS}
    resamples = {c: rng.choice(ix, (draws, len(ix))) for c, ix in strata.items()}
    denom = {c: reference_error[indices].mean(axis=1).clip(1e-9) for c, indices in resamples.items()}
    q_true = conditional_mix(truth)
    base = read_submission(baseline, queries)
    base_error = episode_error(array(base))
    base_r = contest_score(base, targets, groups, reference, max_points=points)["mean_loss_ratio"]
    report = {}
    for name, path in {"control": baseline, **forecasts}.items():
        prediction = read_submission(path, queries)
        score = contest_score(prediction, targets, groups, reference, max_points=points)
        p = array(prediction)
        diff = episode_error(p) - base_error
        sampled_delta = np.mean([diff[ix].mean(axis=1) / denom[c] for c, ix in resamples.items()], axis=0)
        size_sse = np.square(conditional_mix(p) - q_true).sum(axis=1)
        report[name] = {
            "path": str(path),
            "sha256": sha256(path),
            "points": score["technical_points"],
            "R": score["mean_loss_ratio"],
            "delta_R": score["mean_loss_ratio"] - base_r,
            "delta_R_95": np.quantile(sampled_delta, [0.025, 0.975]).tolist(),
            "condition_R": {c: row["loss_ratio"] for c, row in score["by_condition"].items()},
            "p0_mse": float(np.square(p[:, 0] - truth[:, 0]).mean()),
            "conditional_payout_sse": float(size_sse.mean()),
            "win_mass_weighted_conditional_payout_sse": float(np.mean((1 - truth[:, 0]) ** 2 * size_sse)),
        }
    return {
        "data": str(data),
        "episodes": len(episode_counts),
        "queries": len(ids),
        "points_max": points,
        "bootstrap_draws": draws,
        "input_hashes": {f: sha256(data / f) for f in ("queries.csv", "targets.csv", "conditions.csv", "reference.csv")},
        "models": report,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("forecast", nargs="*", help="Candidate as name=forecast.csv")
    parser.add_argument("--data", type=Path, default=KIT / "public/validation")
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--points", type=float)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--make-folds", type=Path, help="Write train and validation folds to this CSV and exit.")
    args = parser.parse_args()
    if args.make_folds:
        make_folds(args.data.parent, args.make_folds)
        return
    points = args.points
    if points is None:
        points = json.loads((args.data.parent / "metadata.json").read_text())["max_points"]
    forecasts = {}
    for item in args.forecast:
        name, separator, path = item.partition("=")
        if not separator or not name or not path or name in forecasts or name == "control":
            parser.error(f"invalid or duplicate name=path candidate {item}")
        forecasts[name] = Path(path)
    output = json.dumps(compare(args.data, args.baseline, forecasts, points, args.draws), indent=2) + "\n"
    if args.out:
        args.out.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
