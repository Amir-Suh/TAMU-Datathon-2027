"""Evaluate prospective lagged peer corrections on train folds and tuning only."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import platform
import shlex
import sys
from time import perf_counter

import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_limits

from run_model_comparison import DEFAULT_COMPARISON, DEFAULT_PUBLIC, _conditions, _outcome_losses, _train_folds, tuning_data
from slot_forecasting.adaptive import AdaptiveLagForecaster
from slot_forecasting.causal import tick_arrays
from slot_forecasting.contest import contest_score
from slot_forecasting.io import load_inputs, read_labels, read_submission, write_json, write_submission
from slot_forecasting.peer_lags import PeerSettings, adaptive_trace, endpoint_rows, peer_predictions
from slot_forecasting.rounds import VERSION, round_metadata
from slot_forecasting.schema import CONDITIONS, PAYOUTS, ValidationError
from slot_forecasting.scoring import score_predictions
from slot_forecasting.tuning_metrics import paired_ratio_interval

ROOT = Path(__file__).resolve().parent
SETTINGS = tuple(PeerSettings(top, shrink) for top in (1, 2) for shrink in (.35, .7))
THREADS = 2


def fit_adaptive(queries, history, labels, *, episodes=None, rate=48):
    if episodes is not None:
        queries = tuple(q for q in queries if q.episode_id in episodes)
        history = tuple(p for p in history if p.episode_id in episodes)
        labels = {q.query_id: labels[q.query_id] for q in queries}
    return AdaptiveLagForecaster(weight_rate=rate).fit(queries, history, labels)


def fold_losses(prediction, outcomes, episode_ids, is_cutoff):
    return {
        part: _outcome_losses(prediction[mask], outcomes[mask], episode_ids[mask])
        for part, mask in (("historical", ~is_cutoff), ("cutoffs", is_cutoff))
    }


def train_cv(queries, history, labels, groups, *, rate=48):
    folds = list(_train_folds(queries, groups))
    episodes = {q.episode_id for q in queries}
    oof_losses = defaultdict(list)
    for held_out in folds:
        model = fit_adaptive(queries, history, labels, episodes=episodes - held_out, rate=rate)
        held_queries = tuple(q for q in queries if q.episode_id in held_out)
        held_history = tuple(p for p in history if p.episode_id in held_out)
        spins, cutoffs = tick_arrays(held_queries, held_history)
        trace = adaptive_trace(held_queries, spins, model)
        machines, ticks = endpoint_rows(spins, cutoffs)
        is_cutoff = ticks == cutoffs[machines]
        outcomes = spins[machines, ticks].copy()
        outcomes[is_cutoff] = [PAYOUTS.index(labels[q.query_id]) for q in held_queries]
        row_episodes = np.array([q.episode_id for q in held_queries])[machines]
        prediction, _ = peer_predictions(held_queries, spins, trace, machines, ticks, SETTINGS)
        oof_losses["self_only"].append(fold_losses(trace.base[machines, ticks], outcomes, row_episodes, is_cutoff))
        for i, config in enumerate(SETTINGS):
            for variant, probabilities in prediction[config].items():
                oof_losses[f"{i}:{variant}"].append(fold_losses(probabilities, outcomes, row_episodes, is_cutoff))
    results = {key: {"folds": scores, "mean": {
        part: {loss: float(np.mean([fold[part][loss] for fold in scores])) for loss in ("brier", "log_loss")}
        for part in ("historical", "cutoffs")}} for key, scores in oof_losses.items()}
    best = min(range(len(SETTINGS)), key=lambda i: results[f"{i}:individual"]["mean"]["cutoffs"]["brier"])
    return SETTINGS[best], {"selected_index": best, "held_out_episodes": [sorted(fold) for fold in folds],
                            "results": results}


def tuning_set(public, metadata):
    queries, history, targets, groups, reference = tuning_data(public / "validation")
    expected = len(CONDITIONS) * ((metadata["counts_per_condition"]["validation"] + 1) // 2)
    episodes = {q.episode_id for q in queries}
    if len(episodes) != expected or len(queries) != expected * metadata["machines_per_episode"]:
        raise ValidationError("Unexpected tuning episode or query count.")
    return queries, history, targets, groups, reference


def matrix_predictions(queries, matrix):
    return {q.query_id: row for q, row in zip(queries, matrix)}


def max_difference(first, second):
    return max(abs(a - b) for key in first for a, b in zip(first[key], second[key]))


def score_tuning(prediction, targets, groups, reference, metadata):
    score = contest_score(prediction, targets, groups, reference,
                          max_points=metadata["max_points"], version=VERSION)
    diagnostics = score_predictions(prediction, targets)
    by_episode = defaultdict(list)
    for target in targets:
        by_episode[target.episode_id].append((prediction[target.query_id][0] - target.probabilities[0]) ** 2)
    p0 = float(np.mean([np.mean(values) for values in by_episode.values()]))
    score["p0_squared_error"] = p0
    score["p0_error_fraction"] = p0 / diagnostics["probability_error"]
    score["win_calibration_bins"] = diagnostics["calibration_bins"]
    score["p0_quantiles_10_90"] = np.quantile([p[0] for p in prediction.values()], [.1, .9]).tolist()
    return score


def save_scores(output, forecasts, filenames, targets, groups, reference, metadata, *, compare_run=None):
    scores, hashes, replay_differences = {}, {}, {}
    for name, forecast in forecasts.items():
        filename = filenames.get(name, f"{name}.csv")
        path = output / filename
        write_submission(path, forecast)
        forecasts[name] = read_submission(path, groups)
        scores[name] = score_tuning(forecasts[name], targets, groups, reference, metadata)
        hashes[filename] = sha256(path.read_bytes()).hexdigest()
        if compare_run is not None:
            previous = read_submission(compare_run / filename, groups)
            replay_differences[name] = max_difference(forecasts[name], previous)
            if replay_differences[name] > 1e-8:
                raise ValidationError(f"Nonreproducible forecast for {name}: {replay_differences[name]}")
    return scores, hashes, replay_differences


def matched_control(metadata, default, queries, computed, override):
    path = override or (DEFAULT_COMPARISON / "adaptive_lag.csv" if metadata["round"] == "main" else None)
    if path is None:
        return computed, 0.0
    control = read_submission(path, {q.query_id for q in queries})
    delta = max_difference(computed, control)
    if delta > 1e-12:
        raise ValidationError(f"The adaptive trace must reproduce the {default} control, got {delta}.")
    return control, delta


def package_versions():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "scikit-learn": sklearn.__version__}


def verify_control(metadata, score):
    if metadata["round"] == "main" and abs(score["technical_points"] - 33.224516) >= 5e-7:
        raise ValidationError("The matched main tuning control did not reproduce 33.224516.")


def summarize_selection(rows, groups):
    summaries = {}
    for condition in ("all", *CONDITIONS):
        selected = [row for row in rows if condition == "all" or groups[row["query_id"]] == condition]
        pairs = [pairs for row in selected for pairs in row["selected_pairs"]]
        lag_counts, signs = Counter(), Counter()
        for row in selected:
            for lag, sign in zip(row["selected_lags"], row["selected_signs"]):
                lag_counts[str(lag)] += 1
                signs[str(sign)] += 1
        summaries[condition] = {
            "rows": len(selected), "selected_rows": sum(bool(row["selected_lags"]) for row in selected),
            "selected_lag_counts": dict(lag_counts), "selected_sign_counts": dict(signs),
            "matched_candidates_mean": float(np.mean([row["matched_candidates"] for row in selected])),
            "supported_candidates_mean": float(np.mean([row["supported_candidates"] for row in selected])),
            "selected_pair_quantiles_0_50_100": np.quantile(pairs, [0, .5, 1]).tolist() if pairs else [],
        }
    return summaries


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
        config, cv = train_cv(train_q, train_h, labels, train_groups, rate=rate)
        model = fit_adaptive(train_q, train_h, labels, rate=rate)
        fit_seconds = perf_counter() - start
        queries, history, targets, groups, reference = tuning_set(public, metadata)
        start = perf_counter()
        spins, cutoffs = tick_arrays(queries, history)
        machines = np.arange(len(queries))
        trace = adaptive_trace(queries, spins, model)
        matrices, diagnostics = peer_predictions(queries, spins, trace, machines, cutoffs, (config,))
        prediction_seconds = perf_counter() - start
    computed = matrix_predictions(queries, trace.base[machines, cutoffs])
    adaptive, control_delta = matched_control(metadata, f"adaptive rate {rate}", queries, computed, control)
    forecasts = {"self_only": adaptive, **{name: matrix_predictions(queries, values)
                                          for name, values in matrices[config].items()}}
    if rate != 48 and metadata["round"] == "main":
        forecasts["adaptive_rate_48"] = read_submission(DEFAULT_COMPARISON / "adaptive_lag.csv", groups)
    output.mkdir(parents=True)
    scores, hashes, differences = save_scores(output, forecasts, {"individual": "tuning_predictions.csv"},
                                              targets, groups, reference, metadata, compare_run=compare_run)
    verify_control(metadata, scores.get("adaptive_rate_48", scores["self_only"]))
    interval = paired_ratio_interval(forecasts["individual"], forecasts["self_only"], targets, groups, reference)
    placebo = paired_ratio_interval(forecasts["aligned_matched"], forecasts["shifted_matched"],
                                     targets, groups, reference)
    write_json(output / "peer_selection.json", diagnostics[config])
    write_json(output / "train_folds.json", cv)
    config_data = {"candidates": [asdict(settings) for settings in SETTINGS], "selected": asdict(config),
                   "adaptive": {"weight_rate": rate, "half_lives": [12, 48], "strength": 8},
                   "peer_lags": list(range(1, 9)), "placebo_shift": 11, "centering_prior_strength": 8,
                   "half_life": 24, "ridge": 2, "coefficient_bounds": [-2, 2], "min_pairs": 8,
                   "evidence_ticks": 12, "historical_endpoints_per_machine": 32, "fold_seed": 314159,
                   "threads": THREADS, "versions": package_versions(), "command": shlex.join([sys.executable, *sys.argv])}
    write_json(output / "config.json", config_data)
    for filename in ("config.json", "train_folds.json", "peer_selection.json"):
        hashes[filename] = sha256((output / filename).read_bytes()).hexdigest()
    deltas = {condition: scores["individual"]["by_condition"][condition]["loss_ratio"] -
                         scores["self_only"]["by_condition"][condition]["loss_ratio"] for condition in CONDITIONS}
    report = {
        "round": metadata["round"], "season_id": metadata["season_id"], "split": "tuning",
        "tuning_episodes": sorted({q.episode_id for q in queries}), "tuning_queries": len(queries),
        "fit_seconds": fit_seconds, "prediction_seconds": prediction_seconds,
        "scores": scores, "paired_vs_adaptive": interval, "paired_aligned_vs_shifted": placebo,
        "shortlist": interval["delta_r"] <= -.01 and max(deltas.values()) <= .05,
        "individual_condition_delta_r": deltas,
        "selected": {name: summarize_selection(rows, groups) for name, rows in diagnostics[config].items()},
        "checks": {"adaptive_control_max_delta": control_delta,
                   "valid_normalized_probabilities_and_query_ids": True,
                   "normalized_replay_max_deltas": differences},
        "true_p0_quantiles_10_90": np.quantile([target.probabilities[0] for target in targets], [.1, .9]).tolist(),
        "sha256": hashes,
    }
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
