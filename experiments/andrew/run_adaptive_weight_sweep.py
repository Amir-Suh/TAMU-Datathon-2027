"""Compare adaptive lag weight rates using train folds and official tuning scores."""

import argparse
import hashlib
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
import sklearn
import threadpoolctl

from run_model_comparison import DEFAULT_COMPARISON, DEFAULT_PUBLIC, _conditions, _outcome_losses, _train_folds, tuning_data
from slot_forecasting.adaptive import AdaptiveLagForecaster
from slot_forecasting.contest import contest_score
from slot_forecasting.io import (
    load_inputs, read_labels, read_submission, require_ids, write_json, write_submission,
)
from slot_forecasting.rounds import VERSION, round_metadata
from slot_forecasting.schema import PAYOUTS, ValidationError
from slot_forecasting.tuning_metrics import paired_ratio_interval


HALF_LIVES = (12, 48)
STRENGTH = 8
CONTROL = 48
EXPECTED_MAIN_POINTS = 33.22451619154785
INITIAL_RATES = (0, 12, 24, 48, 72, 96, 144, 192, 288, 384)
FOLLOW_UP_RATES = (512, 768, 1024, 1536, 2048, 3072)
RATES = INITIAL_RATES + FOLLOW_UP_RATES
SEED = 314159
THREADS = 2


def fit(queries, history, labels):
    return AdaptiveLagForecaster(
        half_lives=HALF_LIVES, strength=STRENGTH, weight_rate=CONTROL,
    ).fit(queries, history, labels)


def losses(queries, labels, prediction):
    require_ids(prediction, {q.query_id for q in queries})
    return _outcome_losses(
        np.array([prediction[q.query_id] for q in queries]),
        np.array([PAYOUTS.index(labels[q.query_id]) for q in queries]),
        [q.episode_id for q in queries],
    )


def cross_validate(queries, history, labels, groups):
    folds = list(_train_folds(queries, groups))
    oof = {rate: {} for rate in RATES}
    by_fold = {rate: [] for rate in RATES}
    for fold in folds:
        train_queries = tuple(q for q in queries if q.episode_id not in fold)
        train_history = tuple(play for play in history if play.episode_id not in fold)
        held_queries = tuple(q for q in queries if q.episode_id in fold)
        held_history = tuple(play for play in history if play.episode_id in fold)
        model = fit(train_queries, train_history, {q.query_id: labels[q.query_id] for q in train_queries})
        for rate in RATES:
            model.weight_rate = rate
            prediction = model.predict(held_queries, held_history)
            by_fold[rate].append(losses(held_queries, labels, prediction))
            oof[rate].update(prediction)
    metrics = {rate: {**losses(queries, labels, oof[rate]), "by_fold": by_fold[rate]} for rate in RATES}
    return folds, metrics


def max_difference(left, right):
    require_ids(left, right)
    return max(abs(a - b) for key in left for a, b in zip(left[key], right[key]))


def p_zero_error(prediction, targets):
    errors = np.square([np.subtract(prediction[t.query_id], t.probabilities) for t in targets])
    return {"mean_squared_error": float(errors[:, 0].mean()),
            "share_of_raw_squared_error": float(errors[:, 0].sum() / errors.sum())}


def prediction_file(directory, rate):
    return directory / f"rate_{rate}.csv"


def sweep(data, output, saved_control_path, compare_to):
    started = perf_counter()
    public, output = data.resolve(), output.resolve()
    if output.exists() or output == public or public in output.parents:
        raise ValidationError("Choose a fresh output directory outside the public data.")
    metadata = round_metadata(public)
    train = public / "train"
    queries, history = load_inputs(train)
    labels = read_labels(train / "labels.csv", queries)
    groups = _conditions(train, queries)
    cv_start = perf_counter()
    folds, cv = cross_validate(queries, history, labels, groups)
    cv_seconds = perf_counter() - cv_start
    selected = min(RATES, key=lambda rate: (cv[rate]["brier"], rate))
    initial_selected = min(INITIAL_RATES, key=lambda rate: (cv[rate]["brier"], rate))
    follow_up_selected = min(FOLLOW_UP_RATES, key=lambda rate: (cv[rate]["brier"], rate))

    tuning_q, tuning_h, targets, conditions, reference = tuning_data(public / "validation")
    tuning_episodes = sorted({q.episode_id for q in tuning_q})
    expected_episodes = len(metadata["conditions"]) * ((metadata["counts_per_condition"]["validation"] + 1) // 2)
    expected_queries = expected_episodes * metadata["machines_per_episode"]
    if (len(tuning_episodes), len(tuning_q)) != (expected_episodes, expected_queries):
        raise ValidationError("The tuning manifest does not match the release metadata.")

    def score(prediction):
        return contest_score(prediction, targets, conditions, reference,
                             max_points=metadata["max_points"], version=VERSION)

    saved_control = read_submission(saved_control_path, conditions)
    saved_score = score(saved_control)
    if metadata["round"] == "main" and abs(saved_score["technical_points"] - EXPECTED_MAIN_POINTS) > 1e-8:
        raise ValidationError("The matched saved control must reproduce 33.224516 on main.")

    output.mkdir(parents=True)
    fit_start = perf_counter()
    model = fit(queries, history, labels)
    fit_seconds = perf_counter() - fit_start
    predictions, results, replay_deltas = {}, {}, {}
    for rate in RATES:
        model.weight_rate = rate
        predict_start = perf_counter()
        variants = model.predict_variants(tuning_q, tuning_h)
        seconds = perf_counter() - predict_start
        prediction = variants["adaptive_lag"]
        if rate == 0:
            fixed_difference = max_difference(prediction, variants["lag_fixed"])
        path = prediction_file(output, rate)
        write_submission(path, prediction)
        predictions[rate] = read_submission(path, conditions)
        results[rate] = {"weight_rate": rate, "train_cutoffs": cv[rate], "tuning": score(predictions[rate]),
                         "tuning_p_zero": p_zero_error(predictions[rate], targets), "prediction_seconds": seconds}
        if compare_to is not None:
            replay = read_submission(prediction_file(compare_to, rate), conditions)
            replay_deltas[rate] = max_difference(predictions[rate], replay)

    control_difference = max_difference(predictions[CONTROL], saved_control)
    if max(control_difference, fixed_difference) > 1e-12:
        raise ValidationError("The rate-48 and rate-0 controls must reproduce their matched forecasts.")
    if replay_deltas and max(replay_deltas.values()) > 1e-8:
        raise ValidationError("The normalized tuning predictions differ from the independent repeat.")

    tuning_best = min(RATES, key=lambda rate: results[rate]["tuning"]["mean_loss_ratio"])
    initial_best = min(INITIAL_RATES, key=lambda rate: results[rate]["tuning"]["mean_loss_ratio"])
    follow_up_best = min(FOLLOW_UP_RATES, key=lambda rate: results[rate]["tuning"]["mean_loss_ratio"])
    for row in results.values():
        row["tuning"]["points_vs_rate_48"] = row["tuning"]["technical_points"] - saved_score["technical_points"]
        row["tuning"]["delta_r_vs_rate_48"] = row["tuning"]["mean_loss_ratio"] - saved_score["mean_loss_ratio"]

    def interval(rate, control_rate):
        return paired_ratio_interval(predictions[rate], predictions[control_rate], targets, conditions, reference,
                                     repeats=2000, seed=SEED)

    leading = sorted({initial_selected, initial_best, selected, tuning_best})
    selected_path = output / "tuning_predictions.csv"
    selected_path.write_bytes(prediction_file(output, selected).read_bytes())
    report = {
        "round": metadata["round"], "season_id": metadata["season_id"], "split": "tuning",
        "tuning_episodes": tuning_episodes, "tuning_queries": len(tuning_q),
        "settings": {"half_lives": HALF_LIVES, "strength": STRENGTH, "control_weight_rate": CONTROL,
                     "initial_grid": INITIAL_RATES, "follow_up_grid": FOLLOW_UP_RATES, "threads": THREADS},
        "follow_up_reason": "The initial grid's train-selected rate was its largest candidate, 384.",
        "selection_rule": "Lowest episode-weighted observed cutoff Brier across four grouped train folds.",
        "fold_seed": SEED, "held_out_episodes_by_train_fold": [sorted(fold) for fold in folds],
        "training_cutoff_labels": len(labels),
        "initial_train_selected_rate": initial_selected, "initial_tuning_best_rate": initial_best,
        "follow_up_train_selected_rate": follow_up_selected, "follow_up_tuning_best_rate": follow_up_best,
        "train_selected_rate": selected, "tuning_best_rate": tuning_best,
        "results": results, "saved_control_tuning": saved_score,
        "paired_delta_r_vs_rate_48": {rate: interval(rate, CONTROL) for rate in leading},
        "paired_delta_r_vs_initial_train_selection": {
            rate: interval(rate, initial_selected) for rate in sorted({selected, tuning_best} - {initial_selected})},
        "checks": {"rate_48_vs_saved_control_max_difference": control_difference,
                   "rate_0_vs_fixed_max_difference": fixed_difference,
                   "repeat_normalized_max_difference_by_rate": replay_deltas,
                   "all_csvs_validated": True},
        "fit_seconds": fit_seconds, "cv_seconds": cv_seconds, "total_seconds": perf_counter() - started,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                     "scikit-learn": sklearn.__version__, "threadpoolctl": threadpoolctl.__version__},
        "prediction_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sorted(output.glob("*.csv"))},
        "saved_control_sha256": hashlib.sha256(saved_control_path.read_bytes()).hexdigest(),
    }
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--saved-control", type=Path, default=DEFAULT_COMPARISON / "adaptive_lag.csv")
    parser.add_argument("--compare-to", type=Path, help="Independent run directory for reproducibility checks.")
    args = parser.parse_args()
    with threadpoolctl.threadpool_limits(limits=THREADS), np.errstate(over="raise", invalid="raise", divide="raise"):
        report = sweep(args.data, args.out, args.saved_control, args.compare_to)
    for rate, row in report["results"].items():
        print(f"{rate:4d}  train Brier {row['train_cutoffs']['brier']:.6f}  "
              f"tuning {row['tuning']['technical_points']:.6f}  R {row['tuning']['mean_loss_ratio']:.6f}")
    print(f"Train-selected rate {report['train_selected_rate']}; tuning best {report['tuning_best_rate']}.")
    print(args.out / "report.json")


if __name__ == "__main__":
    main()
