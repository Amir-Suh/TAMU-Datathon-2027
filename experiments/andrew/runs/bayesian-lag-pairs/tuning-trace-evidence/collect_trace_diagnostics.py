"""Recreate the trace evidence for the Bayesian lag-pair tuning plan."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from run_model_comparison import DEFAULT_PUBLIC, ROOT, _conditions, _train_folds, tuning_data
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.bayesian_lag_pairs import BayesianLagPairForecaster, LagPairConfig, lag_pair_rollout
from slot_forecasting.causal import LAG_PAIRS, tick_arrays
from slot_forecasting.combine_forecasts import conditional_positive, load_component, with_positive_head
from slot_forecasting.contest import contest_score
from slot_forecasting.io import load_inputs, read_labels, write_json
from slot_forecasting.rounds import VERSION
from slot_forecasting.schema import CONDITIONS


def collect():
    frozen = ROOT / "runs/bayesian-lag-pairs/replay"
    report = json.loads((frozen / "report.json").read_text())
    model = BayesianLagPairForecaster(LagPairConfig(**report["config"]))
    model.prior = FrequencyForecaster.load(frozen / "population_prior.json")
    queries, history, targets, groups, reference = tuning_data(DEFAULT_PUBLIC / "validation")
    ids = [q.query_id for q in queries]
    predictions, digest = load_component(frozen / "standalone.csv", ids)
    assert digest == report["sha256"]["standalone"]
    truths = {target.query_id: target.probabilities for target in targets}
    truth = np.asarray([truths[key] for key in ids])
    fallback = np.asarray(model.prior.global_prior[1:])
    fallback = fallback / fallback.sum()
    candidates = {
        "control": predictions,
        "diagnostic_true_win": with_positive_head(truth[:, 0], conditional_positive(predictions, fallback)),
        "diagnostic_true_payout": with_positive_head(predictions[:, 0], conditional_positive(truth, fallback)),
    }
    scores = {key: contest_score(dict(zip(ids, value)), targets, groups, reference, max_points=60, version=VERSION)
              for key, value in candidates.items()}

    spins, cutoffs = tick_arrays(queries, history)
    priors = model.prior_rows(queries)
    snapshots = {}
    for offset in (0, 4, 8, 16, 32):
        _, diagnostic = lag_pair_rollout(spins, priors, model.config, diagnostic_ticks=cutoffs - offset)
        snapshots[offset] = diagnostic["pair_posterior"]
    top = snapshots[0].argmax(axis=1)
    changed = top != snapshots[16].argmax(axis=1)
    by_condition = {}
    labels = np.asarray([groups[key] for key in ids])
    for condition in CONDITIONS:
        mask = labels == condition
        by_condition[condition] = {
            "queries": int(mask.sum()),
            "pair_argmax_changes_from_cutoff_minus_16": int(changed[mask].sum()),
            "mean_pair_mass_by_cutoff_offset": {
                offset: float(value[mask].sum(axis=1).mean()) for offset, value in snapshots.items()
            },
        }

    train = DEFAULT_PUBLIC / "train"
    train_queries, train_history = load_inputs(train)
    train_labels = read_labels(train / "labels.csv", train_queries)
    train_groups = _conditions(train, train_queries)
    pair_counts = {condition: Counter() for condition in CONDITIONS}
    for holdout in _train_folds(train_queries, train_groups):
        fit = tuple(q for q in train_queries if q.episode_id not in holdout)
        held = tuple(q for q in train_queries if q.episode_id in holdout)
        model.fit(fit, (row for row in train_history if row.episode_id not in holdout),
                  {q.query_id: train_labels[q.query_id] for q in fit})
        spins, cutoffs = tick_arrays(held, (row for row in train_history if row.episode_id in holdout))
        _, diagnostic = lag_pair_rollout(spins, model.prior_rows(held), model.config, diagnostic_ticks=cutoffs)
        for query, pair in zip(held, diagnostic["pair_posterior"].argmax(axis=1)):
            a, b = LAG_PAIRS[pair]
            pair_counts[train_groups[query.query_id]][f"{a},{b}"] += 1
    return {
        "split": "tuning", "tuning_queries": len(ids), "control_sha256": digest,
        "tuning_episodes": sorted({q.episode_id for q in queries}), "diagnostic_scores": scores,
        "tuning_trace_by_condition": by_condition, "training_out_of_fold_pair_argmax_counts": pair_counts,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    with threadpool_limits(limits=2):
        write_json(args.out, collect())
