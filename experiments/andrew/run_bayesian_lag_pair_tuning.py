"""Select lag-pair tuning candidates on train, freeze forecasts, then score tuning."""

import argparse
from collections import Counter
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import platform
from time import perf_counter

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from run_model_comparison import DEFAULT_COMPARISON, DEFAULT_PUBLIC, ROOT, _conditions, _outcome_losses, tuning_data
from slot_forecasting.bayesian_pair_tuning import (
    BASE_HAZARD, EXPANDED_GRID, HEADS, Endpoints, TrainingData, WinSettings, fit_pair_tuning, joint_predictions,
)
from slot_forecasting.causal import LAG_PAIRS
from slot_forecasting.combine_forecasts import load_component
from slot_forecasting.contest import contest_score
from slot_forecasting.experiments import episode_folds
from slot_forecasting.io import load_inputs, read_history, read_labels, read_queries, write_json, write_submission
from slot_forecasting.rounds import VERSION, round_metadata
from slot_forecasting.schema import CONDITIONS, ValidationError
from slot_forecasting.scoring import score_predictions
from slot_forecasting.tuning_metrics import paired_ratio_interval

SEED = 314159
MIN_TRAIN_GAIN = .001
TIE = 1e-6
CONTROL = ROOT / "runs/bayesian-lag-pairs/replay/standalone.csv"
CONTROL_HASH = "0e0c1caffa6d0fe5f0e7d39645b8b972d28b84fbc21ab90646f243ae1512a006"
CONTROL_RATIO = .12395840334487744
LOGISTIC = ROOT / "models/logistic_regression/two_stage_self.joblib"


def grouped_folds(queries, groups, n_splits):
    episode_conditions = {}
    for q in queries:
        condition = groups[q.query_id]
        if episode_conditions.setdefault(q.episode_id, condition) != condition:
            raise ValidationError("Condition must be constant within an episode.")
    episodes = np.array(sorted(episode_conditions))
    conditions = [episode_conditions[e] for e in episodes]
    split = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return [set(episodes[held]) for _, held in split.split(episodes, conditions)]


def _subset(data, episodes):
    queries = tuple(q for q in data.queries if q.episode_id in episodes)
    return TrainingData(queries, tuple(p for p in data.history if p.episode_id in episodes),
                        {q.query_id: data.labels[q.query_id] for q in queries})


def partition(data, holdout):
    all_episodes = {q.episode_id for q in data.queries}
    return _subset(data, all_episodes - holdout), _subset(data, holdout)


def _win_metrics(win, outcomes, episodes):
    metrics = _outcome_losses(np.column_stack((1 - win, win)), (outcomes > 0).astype(int), episodes)
    metrics["brier"] *= .5
    return metrics


def _average(fold_metrics, counts):
    result = {}
    for key in fold_metrics[0]:
        present = [(m[key], count) for m, count in zip(fold_metrics, counts) if m[key] is not None]
        result[key] = float(np.average([value for value, _ in present],
                                            weights=[weight for _, weight in present])) if present else None
    return result


def _joint_metrics(forecast, outcomes, episodes):
    metrics = _outcome_losses(forecast, outcomes, episodes)
    positive = outcomes > 0
    if positive.any():
        head = forecast[positive, 1:]
        head = head / head.sum(axis=1, keepdims=True)
        metrics["conditional_positive_log_loss"] = _outcome_losses(
            head, outcomes[positive] - 1, episodes[positive])["log_loss"]
    else:
        metrics["conditional_positive_log_loss"] = None
    return metrics


def _passes(candidate, baseline):
    return (baseline["brier"] - candidate["brier"] >= MIN_TRAIN_GAIN
            and sum(a["brier"] < b["brier"] for a, b in zip(candidate["folds"], baseline["folds"]))
            > len(candidate["folds"]) / 2)


def _choose(scores, order):
    baseline = order[0]
    allowed = [baseline, *(key for key in order[1:] if _passes(scores[key], scores[baseline]))]
    best = min(scores[key]["brier"] for key in allowed)
    return next(key for key in allowed if scores[key]["brier"] <= best + TIE)


def select_settings(data, groups, n_splits):
    """Use inner-fit episodes for all pooled state and held-out labels only for scoring."""
    folds = grouped_folds(data.queries, groups, n_splits)
    partitions = []
    for holdout in folds:
        train, held = partition(data, holdout)
        endpoints = Endpoints.build(held.queries, held.history, include_history=False)
        partitions.append((fit_pair_tuning(train), endpoints, endpoints.outcomes(held.labels),
                           endpoints.episode_ids()))
    counts = [len(fold) for fold in folds]
    cache = {}

    def win_score(settings):
        if settings not in cache:
            rows = []
            for fit, endpoints, y, episodes in partitions:
                win, _ = endpoints.win(fit, settings)
                rows.append(_win_metrics(win, y, episodes))
            cache[settings] = {**_average(rows, counts), "folds": rows, "settings": asdict(settings)}
        return cache[settings]

    a = {str(rho): win_score(WinSettings(rho=rho)) for rho in (0., .50, .80)}
    chosen_prior = _choose(a, list(a))
    base = WinSettings(rho=float(chosen_prior))
    b_grid = {
        "control": base, "reset_144": replace(base, hazard=1/144), "reset_64": replace(base, hazard=1/64),
        "expanded": replace(base, expanded=True),
    }
    b = {name: win_score(settings) for name, settings in b_grid.items()}
    best_hazard = _choose(b, ["control", "reset_144", "reset_64"])
    if best_hazard != "control" and _passes(b["expanded"], b["control"]):
        b_grid["combined"] = replace(b_grid[best_hazard], expanded=True)
        b["combined"] = win_score(b_grid["combined"])
    chosen_win = _choose(b, list(b))
    settings = b_grid[chosen_win]

    by_head = {name: [] for name in (*HEADS, "recent_equal_mass")}
    for fit, endpoints, y, episodes in partitions:
        forecasts, _ = joint_predictions(endpoints, fit, settings)
        for name, forecast in forecasts.items():
            by_head[name].append(_joint_metrics(forecast, y, episodes))
    c = {name: {**_average(metrics, counts), "folds": metrics} for name, metrics in by_head.items()}
    chosen_head = _choose(c, list(HEADS))
    return settings, chosen_head, {
        "folds": [sorted(fold) for fold in folds],
        "pair_prior": {"selected": chosen_prior, "candidates": a},
        "win": {"selected": chosen_win, "candidates": b},
        "positive_head": {"selected": chosen_head, "candidates": c},
    }


def _evaluation(forecast, endpoints, labels):
    y = endpoints.outcomes(labels)
    eps = endpoints.episode_ids()
    cutoff = endpoints.is_cutoff
    return {
        name: {"joint": _joint_metrics(forecast[mask], y[mask], eps[mask]),
               "win": _win_metrics(1 - forecast[mask, 0], y[mask], eps[mask])}
        for name, mask in (("cutoffs", cutoff), ("historical", ~cutoff))
    }


def evaluate_outer_fold(data, groups, holdout):
    train, held = partition(data, holdout)
    settings, head, selection = select_settings(train, groups, n_splits=3)
    fit = fit_pair_tuning(train)
    endpoints = Endpoints.build(held.queries, held.history, include_history=True)
    candidate, _ = joint_predictions(endpoints, fit, settings)
    win, _ = endpoints.win(fit, WinSettings())
    prior = endpoints.positive_prior(fit)
    control = np.column_stack((1 - win, win[:, None] * prior))
    predictions = {"control": control, "win_only": candidate["logistic"], "selected": candidate[head]}
    return {
        "held_out_episodes": sorted(holdout), "selection": selection,
        "learned_pair_prior": fit.prior(settings).tolist(),
        "population_prior": fit.model.prior.global_prior,
        "configuration": {"win": asdict(settings), "positive_head": head},
        "metrics": {key: _evaluation(value, endpoints, held.labels) for key, value in predictions.items()},
    }, predictions


def _tuning_inputs(data):
    validation = data / "validation"
    queries = read_queries(validation / "queries.csv")
    groups = _conditions(validation, queries)
    tuning, _ = episode_folds(queries, groups)
    queries = tuple(q for q in queries if q.episode_id in tuning)
    groups = {q.query_id: groups[q.query_id] for q in queries}
    episode_counts = Counter(dict((q.episode_id, groups[q.query_id]) for q in queries).values())
    if len(tuning) != 48 or len(queries) != 384 or episode_counts != {c: 12 for c in CONDITIONS}:
        raise ValidationError("Expected 48 tuning episodes and 384 tuning queries.")
    return queries, read_history(validation / "history.csv", episode_ids=tuning), groups


def _trace_rows(endpoints, trace):
    rows = {}
    for i, query in enumerate(endpoints.queries):
        snapshots = {}
        for offset in (0, 4, 8, 16, 32):
            tick = query.cutoff - offset
            leading = trace["leading_pair"][i, tick]
            snapshots[offset] = {
                "tick": tick, "leading_pair": LAG_PAIRS[leading],
                "pair_mass": float(trace["pair_mass"][i, tick]),
                "leading_pair_mass": float(trace["leading_pair_mass"][i, tick]),
                "pair_entropy": float(trace["pair_entropy"][i, tick]),
                "support": int(trace["leading_pair_support"][i, tick]),
                "amplitude_distribution": trace["leading_pair_amplitude"][i, tick].tolist(),
            }
        rows[query.query_id] = snapshots
    return {"amplitudes": trace["amplitudes"].tolist(), "queries": rows}


def _hash(path):
    return sha256(path.read_bytes()).hexdigest()


def prepare(data, out):
    started = perf_counter()
    if out.exists() or out == data or data in out.parents:
        raise ValidationError("Choose a fresh run directory outside the public data.")
    metadata = round_metadata(data)
    if metadata["round"] != "main":
        raise ValidationError("This experiment compares against the frozen main tuning control.")
    train_dir = data / "train"
    queries, history = load_inputs(train_dir)
    labels = read_labels(train_dir / "labels.csv", queries)
    groups = _conditions(train_dir, queries)
    train = TrainingData(queries, history, labels)
    outer = [evaluate_outer_fold(train, groups, holdout)[0] for holdout in grouped_folds(queries, groups, 4)]
    settings, head, selection = select_settings(train, groups, n_splits=4)
    fit = fit_pair_tuning(train, logistic=joblib.load(LOGISTIC))
    training_seconds = perf_counter() - started

    # Forecast-only preparation. This path has no tuning target or reference input.
    queries, history, _ = _tuning_inputs(data)
    ids = [q.query_id for q in queries]
    endpoints = Endpoints.build(queries, history, include_history=False)
    baseline, control_hash = load_component(CONTROL, ids)
    if control_hash != CONTROL_HASH:
        raise ValidationError("Frozen 53.383 tuning control has changed.")
    positive = endpoints.positive_prior(fit)
    logistic, logistic_hash = load_component(DEFAULT_COMPARISON / "two_stage_self.csv", ids)
    if np.max(np.abs(positive - logistic[:, 1:] / logistic[:, 1:].sum(axis=1, keepdims=True))) > 1e-12:
        raise ValidationError("The saved logistic checkpoint differs from the frozen payout mix.")
    original_win, original_trace = endpoints.win(fit, WinSettings(), record_trace=True)
    original = np.column_stack((1 - original_win, original_win[:, None] * positive))
    control_delta = float(np.max(np.abs(original - baseline)))
    if control_delta > 1e-12:
        raise ValidationError("Default parameters do not reproduce the frozen control.")
    forecasts, trace = joint_predictions(endpoints, fit, settings)
    finalists = {"win_only": forecasts["logistic"]}
    if head != "logistic":
        finalists["win_and_payout"] = forecasts[head]

    out.mkdir(parents=True)
    prediction_hashes = {}
    for name, forecast in finalists.items():
        path = out / f"{name}.csv"
        write_submission(path, dict(zip(ids, forecast)))
        _, prediction_hashes[name] = load_component(path, ids)
    write_json(out / "train_cv.json", {"outer": outer, "final_training_selection": selection})
    write_json(out / "trace.json", _trace_rows(endpoints, trace))
    write_json(out / "control_trace.json", _trace_rows(endpoints, original_trace["trace"]))
    fit.model.prior.save(out / "population_prior.json")
    write_json(out / "freeze.json", {
        "split": "tuning", "season_id": metadata["season_id"], "queries": len(ids),
        "tuning_episodes": sorted({q.episode_id for q in queries}),
        "configuration": {"win": asdict(settings), "positive_head": head, "base_hazard": BASE_HAZARD,
                          "expanded_grid": EXPANDED_GRID, "pair_prior": fit.prior(settings).tolist()},
        "selection": {"min_brier_gain": MIN_TRAIN_GAIN, "requires_majority_fold_gain": True,
                      "tie_tolerance": TIE, "fold_seed": SEED},
        "prediction_sha256": prediction_hashes, "control_sha256": control_hash,
        "logistic_sha256": logistic_hash, "control_replay_max_abs_delta": control_delta,
        "source_sha256": {path: _hash(Path(__file__).resolve().parent / path) for path in (
            "slot_forecasting/bayesian_lag_pairs.py", "slot_forecasting/bayesian_pair_tuning.py",
            "run_bayesian_lag_pair_tuning.py")},
        "python": platform.python_version(), "threads": 2, "training_seconds": training_seconds,
    })
    print(f"Frozen tuning forecasts: {', '.join(finalists)}")
    print(f"Selected win settings: {asdict(settings)}. Selected payout head: {head}.")
    print(out / "freeze.json")


def score(data, out):
    freeze = json.loads((out / "freeze.json").read_bytes())
    if (out / "report.json").exists():
        raise FileExistsError(out / "report.json")
    # The manifest loader used here remains answer-free until hashes have been verified.
    queries, _, _ = _tuning_inputs(data)
    ids = [q.query_id for q in queries]
    if freeze["tuning_episodes"] != sorted({q.episode_id for q in queries}):
        raise ValidationError("Tuning episode manifest changed after freezing.")
    control, control_hash = load_component(CONTROL, ids)
    if control_hash != CONTROL_HASH or control_hash != freeze["control_sha256"]:
        raise ValidationError("Frozen tuning control has changed.")
    predictions = {"control": dict(zip(ids, control))}
    for name, digest in freeze["prediction_sha256"].items():
        array, actual = load_component(out / f"{name}.csv", ids)
        if digest != actual:
            raise ValidationError(f"Frozen prediction changed: {name}.")
        predictions[name] = dict(zip(ids, array))
    _, _, targets, groups, reference = tuning_data(data / "validation")
    metadata = round_metadata(data)
    scores = {name: contest_score(forecast, targets, groups, reference,
                                  max_points=metadata["max_points"], version=VERSION)
              for name, forecast in predictions.items()}
    if abs(scores["control"]["mean_loss_ratio"] - CONTROL_RATIO) > 1e-10:
        raise ValidationError("Expected to reproduce the frozen tuning control score.")
    paired = {name: paired_ratio_interval(forecast, predictions["control"], targets, groups, reference)
              for name, forecast in predictions.items() if name != "control"}
    conditions = {
        name: {c: row["by_condition"][c]["loss_ratio"] - scores["control"]["by_condition"][c]["loss_ratio"]
               for c in CONDITIONS}
        for name, row in scores.items() if name != "control"
    }
    diagnostics = {}
    for name, forecast in predictions.items():
        truth = np.array([t.probabilities for t in targets])
        matrix = np.array([forecast[t.query_id] for t in targets])
        diagnostics[name] = {
            "mean_squared_error_by_payout": np.square(matrix - truth).mean(axis=0).tolist(),
            "win_calibration_bins": score_predictions(forecast, targets)["calibration_bins"],
        }
    trace_summary = {}
    for name, filename in (("control", "control_trace.json"), ("selected_win", "trace.json")):
        trace = json.loads((out / filename).read_bytes())["queries"]
        trace_summary[name] = {}
        for condition in CONDITIONS:
            rows = [trace[key] for key, label in groups.items() if label == condition]
            trace_summary[name][condition] = {
                "leading_pair_changed_since_cutoff_minus_16": sum(
                    row["0"]["leading_pair"] != row["16"]["leading_pair"] for row in rows),
                "mean_pair_mass": float(np.mean([row["0"]["pair_mass"] for row in rows])),
                "median_leading_pair_entropy": float(np.median([row["0"]["pair_entropy"] for row in rows])),
            }
    write_json(out / "report.json", {
        "split": "tuning", "episodes": len(freeze["tuning_episodes"]), "queries": len(ids),
        "configuration": freeze["configuration"], "freeze_sha256": _hash(out / "freeze.json"),
        "scores": scores, "paired_delta_r": paired, "condition_delta_r": conditions,
        "advances": {name: row["delta_r"] <= -.01 and max(conditions[name].values()) <= .05
                     for name, row in paired.items()},
        "diagnostics": diagnostics, "trace_summary": trace_summary,
    })
    for name, row in scores.items():
        print(f"{name:18} {row['technical_points']:.6f} points, R={row['mean_loss_ratio']:.6f}")
    print(out / "report.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "score"))
    parser.add_argument("--data", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        (prepare if args.phase == "prepare" else score)(args.data.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
