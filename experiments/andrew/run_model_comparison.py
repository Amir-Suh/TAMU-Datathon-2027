"""Train on public/train and compare forecasts on the tuning episodes only."""

import argparse
import platform
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from slot_forecasting.adaptive import AdaptiveLagForecaster
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.contest import contest_score
from slot_forecasting.experiments import episode_folds
from slot_forecasting.io import (
    load_inputs, read_history, read_labels, read_queries, read_rows, read_submission,
    read_targets, require_ids, write_json, write_submission,
)
from slot_forecasting.rounds import VERSION, round_metadata
from slot_forecasting.schema import CONDITIONS, PAYOUTS, ValidationError
from slot_forecasting.win_payout import WinPayoutForecaster, training_rows


ROOT = Path(__file__).resolve().parent
DEFAULT_PUBLIC = ROOT.parents[1] / "main-participants-60-40" / "public"
DEFAULT_COMPARISON = ROOT / "runs" / "model-comparison"


def _conditions(directory, queries):
    rows = read_rows(directory / "conditions.csv", ("query_id", "condition"))
    groups = {row["query_id"]: row["condition"] for row in rows}
    require_ids(groups, {q.query_id for q in queries})
    if len(groups) != len(rows) or set(groups.values()) != set(CONDITIONS):
        raise ValidationError("Expected one released condition per query and all four conditions.")
    return groups


def tuning_data(directory):
    queries = read_queries(directory / "queries.csv")
    groups = _conditions(directory, queries)
    tuning, _ = episode_folds(queries, groups)
    queries = tuple(q for q in queries if q.episode_id in tuning)
    episodes = {q.query_id: q.episode_id for q in queries}
    ids = set(episodes)
    history = read_history(directory / "history.csv", episode_ids=tuning)
    targets = read_targets(directory / "targets.csv", query_ids=ids)
    require_ids({t.query_id for t in targets}, ids)
    if any(episodes[t.query_id] != t.episode_id for t in targets):
        raise ValidationError("Tuning target episode IDs must match the query manifest.")
    reference = read_submission(directory / "reference.csv", ids, selection=("query_id", ids))
    return queries, history, targets, {key: groups[key] for key in ids}, reference


def _train_folds(queries, groups):
    condition_by_episode = {}
    for query in queries:
        condition = groups[query.query_id]
        if condition_by_episode.setdefault(query.episode_id, condition) != condition:
            raise ValidationError("Training condition must be constant within an episode.")
    episodes = np.array(sorted(condition_by_episode))
    conditions = [condition_by_episode[episode] for episode in episodes]
    folds = StratifiedKFold(n_splits=4, shuffle=True, random_state=314159)
    for _, held_out in folds.split(episodes, conditions):
        yield set(episodes[held_out])


def _outcome_losses(prediction, labels, episodes):
    realized = prediction[np.arange(len(labels)), labels]
    brier = (prediction ** 2).sum(axis=1) - 2 * realized + 1
    logloss = -np.log(np.maximum(realized, 1e-15))
    _, index = np.unique(episodes, return_inverse=True)
    size = np.bincount(index)
    return {
        "brier": float((np.bincount(index, weights=brier) / size).mean()),
        "log_loss": float((np.bincount(index, weights=logloss) / size).mean()),
    }


def _fit_tabular(queries, history, labels, groups):
    features, outcomes, machines, contexts = training_rows(queries, history, labels)
    self_features = training_rows(queries, history, labels, peers=False)[0]
    query_episodes = np.array([q.episode_id for q in queries])
    episodes = query_episodes[machines]
    is_cutoff = np.arange(len(outcomes)) >= len(outcomes) - len(queries)
    folds = [np.isin(episodes, sorted(ids)) for ids in _train_folds(queries, groups)]
    models, validation = {}, {}
    for name, inputs, peers, direct in (
        ("two_stage_self", self_features, False, False),
        ("two_stage", features, True, False),
        ("direct_five_class", features, True, True),
    ):
        scores = []
        for strength in (.02, .2):
            oof = np.empty((len(outcomes), len(PAYOUTS)))
            for held_out in folds:
                train = ~held_out
                model = WinPayoutForecaster(strength=strength, peers=peers, direct=direct)
                model.fit_rows(inputs[train], outcomes[train], contexts[train])
                oof[held_out] = model.predict_rows(inputs[held_out], contexts[held_out])
            scores.append({
                "C": strength,
                "all_rows": _outcome_losses(oof, outcomes, episodes),
                "cutoffs": _outcome_losses(oof[is_cutoff], outcomes[is_cutoff], episodes[is_cutoff]),
            })
        strength = min(scores, key=lambda row: row["cutoffs"]["brier"])["C"]
        models[name] = WinPayoutForecaster(strength=strength, peers=peers, direct=direct).fit_rows(
            inputs, outcomes, contexts)
        validation[name] = {"selected_C": strength, "candidates": scores}
    return models, validation, int((~is_cutoff).sum())


def run_comparison(data, output):
    public, output = Path(data).resolve(), Path(output).resolve()
    if output.exists() or output == public or public in output.parents:
        raise ValidationError("Choose a fresh output directory outside the public data.")
    metadata = round_metadata(public)
    train = public / "train"
    train_q, train_h = load_inputs(train)
    labels = read_labels(train / "labels.csv", train_q)
    groups = _conditions(train, train_q)
    start = perf_counter()
    baseline = FrequencyForecaster("pooled_recent", prior_strength=8, half_life=12)
    baseline.fit(train_q, train_h, labels)
    adaptive = AdaptiveLagForecaster().fit(train_q, train_h, labels)
    with threadpool_limits(limits=2):
        tabular, cv, n_history = _fit_tabular(train_q, train_h, labels, groups)
    fit_seconds = perf_counter() - start

    tuning_q, tuning_h, targets, conditions, reference = tuning_data(public / "validation")
    predictions = {"baseline": baseline.predict(tuning_q, tuning_h)}
    predictions.update(adaptive.predict_variants(tuning_q, tuning_h))
    with threadpool_limits(limits=2):
        predictions.update({name: model.predict(tuning_q, tuning_h) for name, model in tabular.items()})
        predictions["two_stage_unblended"] = tabular["two_stage"].predict(
            tuning_q, tuning_h, positive_prior_weight=0)
    control_delta = max(abs(a - b) for key in predictions["baseline"]
                        for a, b in zip(predictions["baseline"][key], predictions["lag_no_context"][key]))
    if control_delta > 1e-12:
        raise ValidationError("The matched no-context control must reproduce the 12-tick baseline.")

    output.mkdir(parents=True)
    results = {}
    for name, prediction in predictions.items():
        path = output / f"{name}.csv"
        write_submission(path, prediction)
        results[name] = contest_score(
            read_submission(path, conditions), targets, conditions, reference,
            max_points=metadata["max_points"], version=VERSION)

    baseline.save(output / "baseline.json")
    joblib.dump({"adaptive_lag": adaptive, **tabular}, output / "models.joblib")
    baseline_points = results["baseline"]["technical_points"]
    for row in results.values():
        row["points_vs_baseline"] = row["technical_points"] - baseline_points
    report = {
        "round": metadata["round"],
        "season_id": metadata["season_id"],
        "split": "tuning",
        "tuning_episodes": sorted({q.episode_id for q in tuning_q}),
        "tuning_queries": len(tuning_q),
        "training_cutoff_labels": len(labels),
        "training_history_labels": n_history,
        "adaptive_lag_settings": {"half_lives": adaptive.half_lives, "strength": adaptive.strength,
                                  "weight_rate": adaptive.weight_rate},
        "tabular_train_cv": cv,
        "training_seconds": fit_seconds,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "scikit-learn": sklearn.__version__},
        "comparison": results,
    }
    write_json(output / "comparison.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--out", type=Path, default=DEFAULT_COMPARISON)
    args = parser.parse_args()
    report = run_comparison(args.data, args.out)
    print(f"Tuning only: {len(report['tuning_episodes'])} episodes, {report['tuning_queries']} queries")
    for name, result in sorted(report["comparison"].items(),
                               key=lambda item: item[1]["mean_loss_ratio"]):
        points = result["technical_points"]
        delta = result["points_vs_baseline"]
        print(f"{name:24} {points:6.3f} points  ({delta:+.3f})  R={result['mean_loss_ratio']:.5f}")
    print(f"Report: {args.out / 'comparison.json'}")


if __name__ == "__main__":
    main()
