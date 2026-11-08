"""Evaluate causal fixed transition retrieval using train folds and tuning episodes."""

import argparse
import hashlib
import platform
from pathlib import Path
import resource
from time import perf_counter

import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from run_model_comparison import DEFAULT_COMPARISON, DEFAULT_PUBLIC, _conditions, _outcome_losses, _train_folds, tuning_data
from slot_forecasting.adaptive import AdaptiveLagForecaster
from slot_forecasting.causal import MISSING, lagged_spins
from slot_forecasting.contest import contest_score
from slot_forecasting.io import load_inputs, read_labels, read_submission, write_json, write_submission
from slot_forecasting.rounds import VERSION, round_metadata
from slot_forecasting.schema import PAYOUTS, ValidationError
from slot_forecasting.transition_retrieval import (
    LEARNED_HISTORY, MATCH_HALF_LIFE, MISMATCH_PENALTY, PRIOR_HALF_LIFE, PRIOR_STRENGTH,
    SHRINKAGE, SIMILARITY_BATCH, SIMILARITY_EPOCHS, SIMILARITY_STEP, SIMILARITY_WIDTH, SUPPORT_CAP,
    TransitionRetrievalForecaster, causal_support, machine_rows, retrieve, retrieval_weights,
)
from slot_forecasting.tuning_metrics import paired_ratio_interval


SEED = 314159
HISTORICAL_ENDPOINTS = 32
THREADS = 2


def verify_kernel():
    query = np.array([[1, 0, -1, -1, -1, -1, -1, -1]])
    keys = np.array([[[2, 0, -1, -1, -1, -1, -1, -1], [0, 0, -1, -1, -1, -1, -1, -1]]])
    ages = np.array([[24, 24]])
    wins = np.array([[1, 0]])
    baseline = np.array([.3])
    weights = retrieval_weights(query, keys, ages)
    expected = np.array([[.125, .125 / np.e]])
    np.testing.assert_allclose(weights, expected, atol=1e-15, rtol=0)
    result = retrieve(query, keys, wins, ages, baseline)
    np.testing.assert_allclose(result[0], (expected[0, 0] + 4 * baseline) / (4 + expected.sum()),
                               atol=1e-15, rtol=0)
    np.testing.assert_allclose(result, retrieve(query, keys[:, ::-1], wins[:, ::-1], ages[:, ::-1], baseline),
                               atol=1e-15, rtol=0)
    np.testing.assert_array_equal(retrieve(np.full_like(query, MISSING), keys, wins, ages, baseline)[0], baseline)
    query[0, 0] = MISSING
    np.testing.assert_array_equal(retrieval_weights(query, keys, ages), [[.0625, .0625]])

    spins = np.array([0, MISSING, 1, 0, MISSING, 2, MISSING, 0, 1, 1, 0])
    endpoints = np.array([3, 6, 8])
    lags = lagged_spins(spins[None])[0]
    observed = np.flatnonzero(spins != MISSING)
    support, _, support_ages = causal_support(observed, spins[observed], lags, endpoints)
    for row, tick in enumerate(endpoints):
        np.testing.assert_array_equal(tick - support_ages[row, support_ages[row] > 0], observed[observed < tick])
        for key, age in zip(support[row], support_ages[row]):
            if age > 0:
                earlier = tick - age - np.arange(1, 9)
                np.testing.assert_array_equal(key, np.where(earlier >= 0, spins[earlier], MISSING))
        altered = spins.copy()
        altered[tick:] = len(PAYOUTS) - 1
        prior = np.array([.6, .25, .1, .04, .01])
        before = machine_rows(spins, lags, np.array([tick]), prior)
        after = machine_rows(altered, lagged_spins(altered[None])[0], np.array([tick]), prior)
        for original, modified in zip(before, after):
            np.testing.assert_array_equal(original, modified)
    return {"toy_kernel_and_mask": True, "order_invariance": True, "strict_prefix_and_future_mutation": True}


def cross_validate(queries, history, labels, groups):
    folds = list(_train_folds(queries, groups))
    folds_report = []
    buckets = {"historical": [], "cutoffs": []}
    for held_out in folds:
        train_q = tuple(q for q in queries if q.episode_id not in held_out)
        train_h = tuple(play for play in history if play.episode_id not in held_out)
        held_q = tuple(q for q in queries if q.episode_id in held_out)
        held_h = tuple(play for play in history if play.episode_id in held_out)
        train_labels = {q.query_id: labels[q.query_id] for q in train_q}
        model = TransitionRetrievalForecaster().fit(train_q, train_h, train_labels)
        rows = model.predict_rows(held_q, held_h, historical_endpoints=HISTORICAL_ENDPOINTS)
        episodes = np.array([q.episode_id for q in held_q])[rows.machines]
        cutoff = rows.outcomes == MISSING
        adaptive = AdaptiveLagForecaster()
        adaptive.prior = model.prior
        adaptive_prediction = adaptive.predict(held_q, held_h)
        model.fit_similarity(train_q, train_h, train_labels, seed=SEED)
        learned = model.predict_rows(held_q, held_h, historical_endpoints=HISTORICAL_ENDPOINTS)
        buckets["historical"].append((rows.outcomes[~cutoff], episodes[~cutoff],
                                      rows.prediction[~cutoff], learned.prediction[~cutoff], rows.recency[~cutoff]))
        buckets["cutoffs"].append((np.array([PAYOUTS.index(labels[q.query_id]) for q in held_q]),
                                   episodes[cutoff], rows.prediction[cutoff], learned.prediction[cutoff], rows.recency[cutoff],
                                   np.array([adaptive_prediction[q.query_id] for q in held_q])))
        folds_report.append(sorted(held_out))

    metrics = {}
    for split, batches in buckets.items():
        outcomes, episodes, *forecasts = map(np.concatenate, zip(*batches))
        metrics[split] = {"rows": len(outcomes), **{
            name: _outcome_losses(prediction, outcomes, episodes)
            for name, prediction in zip(("fixed_retrieval", "learned_retrieval", "recency", "adaptive"), forecasts)
        }}
    return folds_report, metrics


def support_report(rows, queries, targets, prediction, control):
    truth = {target.query_id: target.probabilities for target in targets}
    truth = np.array([truth[q.query_id] for q in queries])
    forecasts = np.array([prediction[q.query_id] for q in queries])
    control_forecasts = np.array([control[q.query_id] for q in queries])
    errors = np.square(forecasts - truth)
    control_errors = np.square(control_forecasts - truth)
    lower, upper = np.quantile(rows.match_mass, [1 / 3, 2 / 3])
    subsets = {"low_match_mass": rows.match_mass <= lower,
               "medium_match_mass": (rows.match_mass > lower) & (rows.match_mass <= upper),
               "high_match_mass": rows.match_mass > upper,
               "sparse_last_64_ticks": rows.recent_observations < SUPPORT_CAP,
               "dense_last_64_ticks": rows.recent_observations == SUPPORT_CAP}
    return {"match_mass_tertile_boundaries": [float(lower), float(upper)],
            "p_zero_mean_squared_error": float(errors[:, 0].mean()),
            "p_zero_share_of_raw_squared_error": float(errors[:, 0].sum() / errors.sum()),
            "subsets": {name: {"queries": int(subset.sum()), "candidate_raw_error": float(errors[subset].sum(axis=1).mean()),
                               "adaptive_raw_error": float(control_errors[subset].sum(axis=1).mean()),
                               "mean_match_mass": float(rows.match_mass[subset].mean()),
                               "mean_effective_support": float(rows.effective_support[subset].mean())}
                        for name, subset in subsets.items() if subset.any()}}


def run(data, output, saved_control_path, compare_to):
    started = perf_counter()
    public, output = data.resolve(), output.resolve()
    if output.exists() or output == public or public in output.parents:
        raise ValidationError("Choose a fresh output directory outside the public data.")
    checks = verify_kernel()
    metadata = round_metadata(public)
    train = public / "train"
    queries, history = load_inputs(train)
    labels = read_labels(train / "labels.csv", queries)
    groups = _conditions(train, queries)
    cv_start = perf_counter()
    folds, cv = cross_validate(queries, history, labels, groups)
    cv_seconds = perf_counter() - cv_start
    cutoff = cv["cutoffs"]
    competitive = cutoff["fixed_retrieval"]["brier"] <= cutoff["adaptive"]["brier"] + .005
    methods = ("fixed_retrieval", "learned_retrieval")
    selected = min(methods, key=lambda name: cutoff[name]["brier"])

    fit_start = perf_counter()
    model = TransitionRetrievalForecaster().fit(queries, history, labels)
    fit_seconds = perf_counter() - fit_start
    tuning_q, tuning_h, targets, conditions, reference = tuning_data(public / "validation")
    tuning_episodes = sorted({q.episode_id for q in tuning_q})
    expected_episodes = len(metadata["conditions"]) * ((metadata["counts_per_condition"]["validation"] + 1) // 2)
    if (len(tuning_episodes), len(tuning_q)) != (expected_episodes, expected_episodes * metadata["machines_per_episode"]):
        raise ValidationError("The tuning manifest does not match the release metadata.")
    prediction_start = perf_counter()
    rows = model.predict_rows(tuning_q, tuning_h)
    fixed_seconds = perf_counter() - prediction_start
    learned_fit_start = perf_counter()
    model.fit_similarity(queries, history, labels, seed=SEED)
    learned_fit_seconds = perf_counter() - learned_fit_start
    prediction_start = perf_counter()
    learned_rows = model.predict_rows(tuning_q, tuning_h)
    learned_seconds = perf_counter() - prediction_start
    variants_rows = dict(zip(methods, (rows, learned_rows)))
    saved = read_submission(saved_control_path, conditions)

    def score(prediction):
        return contest_score(prediction, targets, conditions, reference,
                             max_points=metadata["max_points"], version=VERSION)

    saved_score = score(saved)
    if metadata["round"] == "main" and abs(saved_score["technical_points"] - 33.22451619154785) > 1e-8:
        raise ValidationError("The saved adaptive control must reproduce 33.224516 on main.")
    adaptive = AdaptiveLagForecaster()
    adaptive.prior = model.prior
    variants = adaptive.predict_variants(tuning_q, tuning_h)
    adaptive_rows = np.array([saved[q.query_id] for q in tuning_q])
    recency_control = np.array([variants["lag_no_context"][q.query_id] for q in tuning_q])
    control_delta = np.max(np.abs(adaptive_rows - np.array([variants["adaptive_lag"][q.query_id] for q in tuning_q])))
    recency_delta = np.max(np.abs(recency_control - rows.recency))
    adaptive_positive = adaptive_rows[:, 1:] / adaptive_rows[:, 1:].sum(axis=1, keepdims=True)
    payout_delta = max(np.max(np.abs(candidate.prediction[:, 1:]
                                    / candidate.prediction[:, 1:].sum(axis=1, keepdims=True) - adaptive_positive))
                       for candidate in variants_rows.values())
    if max(control_delta, recency_delta, payout_delta) > 1e-12:
        raise ValidationError("Adaptive, recency, or fixed payout head control failed.")
    checks.update(adaptive_saved_max_difference=float(control_delta), recency_max_difference=float(recency_delta),
                  positive_head_max_difference=float(payout_delta))

    output.mkdir(parents=True)
    predictions = {}
    for name, candidate in variants_rows.items():
        path = output / f"{name}.csv"
        write_submission(path, {q.query_id: tuple(row) for q, row in zip(tuning_q, candidate.prediction)})
        predictions[name] = read_submission(path, conditions)
    (output / "tuning_predictions.csv").write_bytes((output / f"{selected}.csv").read_bytes())
    recency = {q.query_id: tuple(row) for q, row in zip(tuning_q, rows.recency)}
    if compare_to is not None:
        for name, prediction in predictions.items():
            replay = read_submission(compare_to / f"{name}.csv", conditions)
            delta = max(abs(a - b) for key in prediction for a, b in zip(prediction[key], replay[key]))
            if delta > 1e-8:
                raise ValidationError("The normalized tuning predictions differ from the independent repeat.")
            checks[f"{name}_repeat_normalized_max_difference"] = delta
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {
        "round": metadata["round"], "season_id": metadata["season_id"], "split": "tuning",
        "tuning_episodes": tuning_episodes, "tuning_queries": len(tuning_q),
        "settings": {"support_cap": SUPPORT_CAP, "match_half_life": MATCH_HALF_LIFE,
                     "mismatch_penalty": MISMATCH_PENALTY, "shrinkage": SHRINKAGE,
                     "prior_half_life": PRIOR_HALF_LIFE, "prior_strength": PRIOR_STRENGTH,
                     "historical_endpoints": HISTORICAL_ENDPOINTS, "threads": THREADS, "fold_seed": SEED,
                     "learned_historical_endpoints": LEARNED_HISTORY, "learned_seed": SEED,
                     "similarity_hidden_width": SIMILARITY_WIDTH, "similarity_epochs": SIMILARITY_EPOCHS,
                     "similarity_batch": SIMILARITY_BATCH, "similarity_adam_step": SIMILARITY_STEP},
        "held_out_episodes_by_train_fold": folds, "training_cv": cv, "train_selected_method": selected,
        "learned_similarity_gate": {"metric": "train cutoff Brier", "allowed_excess_vs_adaptive": .005,
                                    "fixed_is_competitive": competitive, "trained": True},
        "tuning": {"adaptive": saved_score, "recency": score(recency),
                   **{name: score(prediction) for name, prediction in predictions.items()}},
        "paired_delta_r_vs_adaptive": {name: paired_ratio_interval(prediction, saved, targets, conditions, reference)
                                      for name, prediction in predictions.items()},
        "tuning_support_diagnostics": {name: support_report(variants_rows[name], tuning_q, targets, prediction, saved)
                                       for name, prediction in predictions.items()},
        "checks": checks, "cv_seconds": cv_seconds, "prior_fit_seconds": fit_seconds,
        "learned_fit_seconds": learned_fit_seconds, "total_seconds": perf_counter() - started,
        "prediction_seconds": dict(zip(methods, (fixed_seconds, learned_seconds))),
        "process_peak_rss_bytes": rss if platform.system() == "Darwin" else rss * 1024,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scikit-learn": sklearn.__version__},
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
    parser.add_argument("--compare-to", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=THREADS):
        report = run(args.data, args.out, args.saved_control, args.compare_to)
    for name in ("fixed_retrieval", "learned_retrieval"):
        print(f"{name}: {report['tuning'][name]['technical_points']:.6f} tuning points")
    print(args.out / "report.json")


if __name__ == "__main__":
    main()
