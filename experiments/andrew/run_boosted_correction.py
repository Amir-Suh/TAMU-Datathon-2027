"""Fit nested cross-fitted offset corrections and score the tuning split only."""

import argparse
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
from pathlib import Path
import shlex
import sys
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from run_model_comparison import DEFAULT_COMPARISON, DEFAULT_PUBLIC, _conditions, _train_folds
from run_peer_lags import (
    ROOT, THREADS, fit_adaptive, fold_losses, matched_control, matrix_predictions, package_versions,
    save_scores, tuning_set, verify_control,
)
from slot_forecasting.boosted_correction import (
    SEED, BoostInputs, OffsetCalibration, OffsetTrees, build_inputs, corrected_probabilities,
)
from slot_forecasting.causal import feature_rows, tick_arrays
from slot_forecasting.io import load_inputs, read_labels, read_submission, write_json
from slot_forecasting.peer_lags import endpoint_rows
from slot_forecasting.rounds import round_metadata
from slot_forecasting.schema import CONDITIONS, PAYOUTS, ValidationError
from slot_forecasting.tuning_metrics import paired_ratio_interval

TREE_BUDGETS = (30, 60)


@dataclass(frozen=True)
class EndpointSample:
    queries: tuple
    spins: np.ndarray
    machines: np.ndarray
    ticks: np.ndarray
    outcomes: np.ndarray
    episodes: np.ndarray
    is_cutoff: np.ndarray
    local_features: np.ndarray

    def adaptive_inputs(self, model):
        return build_inputs(self.queries, self.spins, self.machines, self.ticks, model,
                            local_features=self.local_features)


def endpoint_sample(queries, history, labels):
    spins, cutoffs = tick_arrays(queries, history)
    machines, ticks = endpoint_rows(spins, cutoffs)
    is_cutoff = ticks == cutoffs[machines]
    outcomes = spins[machines, ticks].copy()
    outcomes[is_cutoff] = [PAYOUTS.index(labels[q.query_id]) for q in queries]
    episodes = np.array([q.episode_id for q in queries])[machines]
    local_features = feature_rows(queries, spins, machines, ticks, peers=False)
    return EndpointSample(queries, spins, machines, ticks, outcomes, episodes, is_cutoff, local_features)


def cross_fitted_inputs(queries, history, labels, folds, *, rate=48):
    """Build each outer holdout and its disjoint inner base forecasts."""
    episode_ids = {q.episode_id for q in queries}
    samples, outer_inputs, inner_inputs = [], [], {}
    for held_out in folds:
        held_queries = tuple(q for q in queries if q.episode_id in held_out)
        held_history = tuple(p for p in history if p.episode_id in held_out)
        sample = endpoint_sample(held_queries, held_history, labels)
        model = fit_adaptive(queries, history, labels, episodes=episode_ids - held_out, rate=rate)
        samples.append(sample)
        outer_inputs.append(sample.adaptive_inputs(model))
    for a, b in combinations(range(len(folds)), 2):
        model = fit_adaptive(queries, history, labels, episodes=episode_ids - folds[a] - folds[b], rate=rate)
        inner_inputs[a, b] = samples[b].adaptive_inputs(model)
        inner_inputs[b, a] = samples[a].adaptive_inputs(model)
    return samples, outer_inputs, inner_inputs


def combine_samples(samples, inputs):
    outcomes = np.concatenate([sample.outcomes for sample in samples])
    episodes = np.concatenate([sample.episodes for sample in samples])
    _, inverse, counts = np.unique(episodes, return_inverse=True, return_counts=True)
    weights = len(episodes) / (len(counts) * counts[inverse])
    return (BoostInputs(np.concatenate([row.features for row in inputs]), np.concatenate([row.offset for row in inputs]),
                        np.concatenate([row.base for row in inputs])), outcomes > 0, weights)


def fit_corrections(inputs, outcomes, weights, *, rounds):
    calibration = OffsetCalibration().fit(inputs.offset, outcomes, weights)
    trees = OffsetTrees(rounds=rounds).fit(inputs.features, inputs.offset, outcomes, weights) if rounds else None
    return calibration, trees


def train_cv(queries, history, labels, groups, *, rate):
    folds = list(_train_folds(queries, groups))
    samples, outer_inputs, inner_inputs = cross_fitted_inputs(queries, history, labels, folds, rate=rate)
    results = {name: [] for name in ("zero_correction", "calibration", *(f"trees_{n}" for n in TREE_BUDGETS))}
    manifest = []
    for outer, sample in enumerate(samples):
        inner_folds = [i for i in range(len(folds)) if i != outer]
        training = combine_samples([samples[i] for i in inner_folds], [inner_inputs[outer, i] for i in inner_folds])
        calibration, trees = fit_corrections(*training, rounds=max(TREE_BUDGETS))
        for name, prediction in corrected_probabilities(outer_inputs[outer], calibration, trees).items():
            results[name].append(fold_losses(prediction, sample.outcomes, sample.episodes, sample.is_cutoff))
        manifest.append({"held_out_episodes": sorted(folds[outer]), "historical_labels": int((~sample.is_cutoff).sum()),
                         "cutoff_labels": int(sample.is_cutoff.sum()), "inner_fold_indices": inner_folds,
                         "calibration_coefficients": calibration.coefficients.tolist()})
    results = {name: {"folds": rows, "mean": {
        part: {metric: float(np.mean([row[part][metric] for row in rows])) for metric in ("brier", "log_loss")}
        for part in ("historical", "cutoffs")}} for name, rows in results.items()}
    lead = min((name for name in results if name != "zero_correction"),
               key=lambda name: results[name]["mean"]["cutoffs"]["brier"])
    tree_budget = min(TREE_BUDGETS, key=lambda n: results[f"trees_{n}"]["mean"]["cutoffs"]["brier"])
    calibration, trees = fit_corrections(*combine_samples(samples, outer_inputs), rounds=max(TREE_BUDGETS))
    report = {"lead_correction": lead, "selected_tree_budget": tree_budget, "fold_manifest": manifest,
              "historical_labels": sum(fold["historical_labels"] for fold in manifest),
              "cutoff_labels": sum(fold["cutoff_labels"] for fold in manifest), "results": results,
              "all_train_calibration_coefficients": calibration.coefficients.tolist()}
    return lead, calibration, trees, report


def run(data, output, *, rate=48, control=None, compare_run=None):
    public, output = Path(data).resolve(), Path(output).resolve()
    if output.exists() or output == public or public in output.parents:
        raise ValidationError("Choose a fresh output directory outside the public data.")
    metadata = round_metadata(public)
    if rate != 48 and metadata["round"] == "main" and control is None:
        raise ValidationError("Supply the saved matched control for a main run with a changed adaptive rate.")
    train = public / "train"
    train_q, train_h = load_inputs(train)
    labels = read_labels(train / "labels.csv", train_q)
    train_groups = _conditions(train, train_q)
    with threadpool_limits(limits=THREADS):
        start = perf_counter()
        lead, calibration, trees, cv = train_cv(train_q, train_h, labels, train_groups, rate=rate)
        model = fit_adaptive(train_q, train_h, labels, rate=rate)
        fit_seconds = perf_counter() - start
        queries, history, targets, groups, reference = tuning_set(public, metadata)
        start = perf_counter()
        spins, cutoffs = tick_arrays(queries, history)
        inputs = build_inputs(queries, spins, np.arange(len(queries)), cutoffs, model)
        matrices = corrected_probabilities(inputs, calibration, trees)
        prediction_seconds = perf_counter() - start
    forecasts = {name: matrix_predictions(queries, values) for name, values in matrices.items()}
    adaptive, control_delta = matched_control(metadata, f"adaptive rate {rate}", queries,
                                              forecasts["zero_correction"], control)
    forecasts["zero_correction"] = adaptive
    if rate != 48 and metadata["round"] == "main":
        forecasts["adaptive_rate_48"] = read_submission(DEFAULT_COMPARISON / "adaptive_lag.csv", groups)
    output.mkdir(parents=True)
    scores, hashes, differences = save_scores(output, forecasts, {lead: "tuning_predictions.csv"},
                                              targets, groups, reference, metadata, compare_run=compare_run)
    verify_control(metadata, scores.get("adaptive_rate_48", scores["zero_correction"]))
    intervals = {name: paired_ratio_interval(forecasts[name], forecasts["zero_correction"], targets, groups, reference)
                 for name in ("calibration", *(f"trees_{n}" for n in TREE_BUDGETS))}
    deltas = {condition: scores[lead]["by_condition"][condition]["loss_ratio"] -
                         scores["zero_correction"]["by_condition"][condition]["loss_ratio"] for condition in CONDITIONS}
    train_winner = min(cv["results"], key=lambda name: cv["results"][name]["mean"]["cutoffs"]["brier"])
    shortlist = train_winner == lead and intervals[lead]["delta_r"] <= -.01 and max(deltas.values()) <= .05
    config = {"adaptive": {"weight_rate": rate, "half_lives": [12, 48], "strength": 8},
              "calibration_ridge": calibration.ridge, "tree_budgets": TREE_BUDGETS,
              "max_depth": trees.max_depth, "min_samples_leaf": trees.min_samples_leaf,
              "shrinkage": trees.shrinkage, "step_limit": trees.step_limit, "curvature_floor": trees.curvature_floor,
              "history_labels_per_machine": 32, "seed": SEED, "threads": THREADS, "versions": package_versions(),
              "lead_correction": lead, "command": shlex.join([sys.executable, *sys.argv])}
    write_json(output / "config.json", config)
    write_json(output / "train_folds.json", cv)
    for filename in ("config.json", "train_folds.json"):
        hashes[filename] = sha256((output / filename).read_bytes()).hexdigest()
    report = {"round": metadata["round"], "season_id": metadata["season_id"], "split": "tuning",
              "tuning_episodes": sorted({q.episode_id for q in queries}), "tuning_queries": len(queries),
              "fit_seconds": fit_seconds, "prediction_seconds": prediction_seconds,
              "scores": scores, "paired_vs_matched_adaptive": intervals, "lead_correction": lead,
              "train_fold_winner": train_winner, "shortlist": shortlist, "lead_condition_delta_r": deltas,
              "checks": {"adaptive_control_max_delta": control_delta, "valid_normalized_probabilities_and_query_ids": True,
                         "normalized_replay_max_deltas": differences}, "sha256": hashes,
              "true_p0_quantiles_10_90": np.quantile([target.probabilities[0] for target in targets], [.1, .9]).tolist()}
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--adaptive-rate", type=float, default=48)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--compare-run", type=Path)
    args = parser.parse_args()
    report = run(args.data, args.out, rate=args.adaptive_rate, control=args.control, compare_run=args.compare_run)
    for name, score in report["scores"].items():
        print(f"{name:18} {score['technical_points']:.6f} points | R = {score['mean_loss_ratio']:.6f}")
    print(args.out / "report.json")


if __name__ == "__main__":
    main()
