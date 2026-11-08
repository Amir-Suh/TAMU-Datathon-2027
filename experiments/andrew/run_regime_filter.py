"""Select causal regime filters on grouped train folds, then score tuning only."""

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import hashlib
from pathlib import Path
import platform
import shlex
import sys
from time import perf_counter

import numpy as np
import scipy
from scipy.special import expit
import sklearn
from threadpoolctl import threadpool_limits

from run_model_comparison import DEFAULT_COMPARISON, DEFAULT_PUBLIC, _conditions, _outcome_losses, _train_folds, tuning_data
from slot_forecasting.adaptive import AdaptiveLagForecaster
from slot_forecasting.causal import MISSING, tick_arrays
from slot_forecasting.contest import contest_score
from slot_forecasting.io import load_inputs, read_labels, read_submission, write_json, write_submission
from slot_forecasting.peer_lags import adaptive_trace, endpoint_rows
from slot_forecasting.regime_filter import (
    REGIME_PRIORS, REGIME_STATES, RegimeConfig, filter_endpoints, filter_win_windows,
)
from slot_forecasting.rounds import VERSION, round_metadata
from slot_forecasting.schema import CONDITIONS, PAYOUTS, Query, ValidationError
from slot_forecasting.tuning_metrics import paired_ratio_interval


ROOT = Path(__file__).resolve().parent
SYNTHETIC_SEED = 161803
BASE = RegimeConfig("base", 0)
GRID = (BASE, *(RegimeConfig(family, hazard) for family in ("level", "direction")
                for hazard in (0, 1 / 64, 1 / 24)))


def _switch_test(rng, config, base_win, no_lag_win, truth, switch, target_state):
    shape = truth.shape
    outcomes = (rng.random(shape) < truth).astype(np.int8)
    base = np.stack((1 - base_win, base_win * .7, base_win * .19, base_win * .1, base_win * .01), axis=-1)
    machines, ticks = np.indices(shape).reshape(2, -1)
    prediction, posterior = filter_endpoints(base, no_lag_win, outcomes, machines, ticks, config)
    win = (1 - prediction[:, 0]).reshape(shape)
    mass = posterior[:, target_state].reshape(shape)
    span = slice(switch + 10, switch + 40)
    brier = float(np.mean((win[:, span] - outcomes[:, span]) ** 2))
    base_brier = float(np.mean((base_win[:, span] - outcomes[:, span]) ** 2))
    lags = [int(crossings[0]) for row in mass
            for crossings in [np.flatnonzero(row[switch:switch + 40] > .5)] if crossings.size]
    assert brier < base_brier and len(lags) > shape[0] // 2
    return {"switch_tick": switch, "detected_within_40_ticks": len(lags), "machines": shape[0],
            "median_detection_lag": float(np.median(lags)), "post_switch_brier": brier,
            "post_switch_base_brier": base_brier}


def _verify_filter(model):
    prior = np.array([.6, .4])
    state = ((1, 0), (1, 1))
    win = np.array([.5, expit(1.)])
    hazard = .25
    p = np.full((1, 3), .5)
    outcome = np.array([[1, MISSING, 0]])
    forecasts, final = filter_win_windows(p, p, outcome, state, prior, hazard)
    post_first = prior * win / (prior @ win)
    pre_second = .75 * post_first + .25 * prior
    pre_third = .75 * pre_second + .25 * prior
    np.testing.assert_allclose(forecasts[0], [prior @ win, pre_second @ win, pre_third @ win], rtol=0, atol=1e-14)
    np.testing.assert_allclose(final[0], pre_third, rtol=0, atol=1e-14)
    p = np.full((2, 41), .4)
    missing = np.full(p.shape, MISSING)
    for config in GRID:
        prior = np.array(REGIME_PRIORS[config.family])
        forecast, posterior = filter_win_windows(p, p, missing, REGIME_STATES[config.family], prior, config.hazard)
        np.testing.assert_allclose(posterior, np.tile(prior, (len(p), 1)), rtol=0, atol=1e-14)
        assert np.isfinite(forecast).all() and ((forecast >= 0) & (forecast <= 1)).all()

    rng = np.random.default_rng(SYNTHETIC_SEED)
    switch, n_ticks, n_machines = 80, 160, 32
    p = np.full((n_machines, n_ticks), .5)
    level_truth = np.where(np.arange(n_ticks)[None, :] < switch, expit(-1), expit(1)) + np.zeros_like(p)
    level = _switch_test(rng, RegimeConfig("level"), p, p, level_truth, switch, 2)
    lag_win = np.where(rng.random(p.shape) < .5, .2, .8)
    sign_truth = np.where(np.arange(n_ticks)[None, :] < switch, lag_win, 1 - lag_win)
    direction = _switch_test(rng, RegimeConfig("direction"), lag_win, p, sign_truth, switch, 4)

    q = tuple(Query(f"q{i}", "episode", f"m{i}", "balanced", 0, 64) for i in range(8))
    spins = rng.integers(MISSING, len(PAYOUTS), size=(len(q), 65))
    change_tick = 32
    changed = spins.copy()
    changed[:, change_tick:] = np.where(changed[:, change_tick:] > 0, 0, 1)
    prior_tick_changed = changed.copy()
    prior_tick_changed[:, change_tick - 1] = np.where(spins[:, change_tick - 1] > 0, 0, 1)
    trace = adaptive_trace(q, spins, model)
    altered_trace = adaptive_trace(q, changed, model)
    prior_altered_trace = adaptive_trace(q, prior_tick_changed, model)
    machines, ticks = np.indices((len(q), change_tick + 1)).reshape(2, -1)
    base_win = 1 - trace.base[..., 0]
    filtered_base = filter_win_windows(base_win, trace.no_lag_win, spins, REGIME_STATES["direction"],
                                       (1., 0., 0., 0., 0.), .1)[0]
    base_delta = float(np.max(np.abs(filtered_base - base_win)))
    assert base_delta <= 1e-12
    for config in GRID:
        ordinary = filter_endpoints(trace.base, trace.no_lag_win, spins, machines, ticks, config)
        mutated = filter_endpoints(altered_trace.base, altered_trace.no_lag_win, changed, machines, ticks, config)
        previous = filter_endpoints(prior_altered_trace.base, prior_altered_trace.no_lag_win,
                                    prior_tick_changed, machines, ticks, config)
        for original, alternative in zip(ordinary, mutated):
            np.testing.assert_array_equal(original, alternative)
        assert np.max(np.abs(ordinary[0][ticks == change_tick] - previous[0][ticks == change_tick])) > 1e-6
    return {"hand_computed_sequence": True, "all_missing_prior_and_normalization": True,
            "base_only_max_win_delta": base_delta, "same_tick_and_future_invariance": True,
            "previous_tick_changes_forecast": True, "seeded_level_switch": level,
            "seeded_sign_reversal": direction}


def _outcomes_at_endpoints(queries, spins, cutoffs, labels):
    machines, ticks = endpoint_rows(spins, cutoffs)
    n_history = len(machines) - len(queries)
    outcomes = np.r_[spins[machines[:n_history], ticks[:n_history]],
                     [PAYOUTS.index(labels[q.query_id]) for q in queries]]
    return machines, ticks, outcomes, n_history


def _support_summary(spins, machines, ticks, groups):
    prefix = np.pad(np.cumsum(spins != MISSING, axis=1), ((0, 0), (1, 0)))
    support = prefix[machines, ticks] - prefix[machines, np.maximum(0, ticks - BASE.window)]
    return {condition: dict(zip(("min", "p10", "p50", "p90", "max"),
                               map(float, np.quantile(support[groups == condition], [0, .1, .5, .9, 1]))))
            for condition in CONDITIONS}


def _cross_validate(queries, history, labels, groups, rates):
    spins, cutoffs = tick_arrays(queries, history)
    machines, ticks, outcomes, n_history = _outcomes_at_endpoints(queries, spins, cutoffs, labels)
    episodes = np.array([q.episode_id for q in queries])
    conditions = np.array([groups[q.query_id] for q in queries])[machines]
    endpoint_episodes = episodes[machines]
    oof = {rate: {config: np.empty((len(outcomes), len(PAYOUTS))) for config in GRID} for rate in rates}
    folds = []
    for held_out in _train_folds(queries, groups):
        folds.append(sorted(held_out))
        fit_q = tuple(q for q in queries if q.episode_id not in held_out)
        fit_h = tuple(p for p in history if p.episode_id not in held_out)
        held_q = tuple(q for q in queries if q.episode_id in held_out)
        fit_labels = {q.query_id: labels[q.query_id] for q in fit_q}
        mask = np.isin(episodes, sorted(held_out))
        held = mask[machines]
        held_machines = (np.cumsum(mask) - 1)[machines[held]]
        held_spins = spins[mask]
        for rate in rates:
            model = AdaptiveLagForecaster(weight_rate=rate).fit(fit_q, fit_h, fit_labels)
            trace = adaptive_trace(held_q, held_spins, model)
            for config in GRID:
                oof[rate][config][held] = filter_endpoints(trace.base, trace.no_lag_win, held_spins,
                                                           held_machines, ticks[held], config)[0]
    scores = {rate: {config: {
        "historical": _outcome_losses(pred[:n_history], outcomes[:n_history], endpoint_episodes[:n_history]),
        "cutoffs": _outcome_losses(pred[n_history:], outcomes[n_history:], endpoint_episodes[n_history:]),
    } for config, pred in candidates.items()} for rate, candidates in oof.items()}
    support = {
        "historical": _support_summary(spins, machines[:n_history], ticks[:n_history], conditions[:n_history]),
        "cutoffs": _support_summary(spins, machines[n_history:], ticks[n_history:], conditions[n_history:]),
    }
    return scores, folds, n_history, support


def _verify_trace(queries, history, trace, adaptive, groups):
    chosen_episodes = {next(q.episode_id for q in queries if groups[q.query_id] == condition)
                       for condition in CONDITIONS}
    indices = [i for i, q in enumerate(queries) if q.episode_id in chosen_episodes]
    by_episode = defaultdict(list)
    for play in history:
        if play.episode_id in chosen_episodes:
            by_episode[play.episode_id].append(play)
    max_delta = 0.
    for offset in (None, 40, 24, 8, 1, 0):
        q = tuple(replace(queries[i], cutoff=0 if offset is None else max(0, queries[i].cutoff - offset))
                  for i in indices)
        episode_cutoffs = {row.episode_id: row.cutoff for row in q}
        h = tuple(p for episode, cutoff in episode_cutoffs.items() for p in by_episode[episode] if p.tick < cutoff)
        exact = adaptive.predict_variants(q, h)
        for i, query in zip(indices, q):
            expected = trace.base[i, query.cutoff]
            no_lag_win = trace.no_lag_win[i, query.cutoff]
            max_delta = max(max_delta, float(np.max(np.abs(expected - exact["adaptive_lag"][query.query_id]))),
                            abs(no_lag_win - (1 - exact["lag_no_context"][query.query_id][0])))
    assert max_delta <= 1e-12
    return {"checked_episodes": sorted(chosen_episodes), "checked_cutoff_offsets": ["tick_zero", 40, 24, 8, 1, 0],
            "max_absolute_probability_delta": max_delta}


def _choose(cv):
    metric = lambda config: cv[config]["cutoffs"]["brier"]
    return {"base": BASE, "fixed_state": min((c for c in cv if c.family != "base" and c.hazard == 0), key=metric),
            "constant_hazard": min((c for c in cv if c.hazard > 0), key=metric)}, min(cv, key=metric)


def _p0_error(predictions, targets, groups):
    by_episode = defaultdict(list)
    condition_by_episode = {}
    for target in targets:
        key = target.query_id
        error = (np.asarray(predictions[key]) - target.probabilities) ** 2
        by_episode[target.episode_id].append(error)
        condition_by_episode[target.episode_id] = groups[key]
    conditions = {condition: np.mean([np.mean(errors, axis=0) for episode, errors in by_episode.items()
                                      if condition_by_episode[episode] == condition], axis=0)
                  for condition in CONDITIONS}
    aggregate = np.mean(list(conditions.values()), axis=0)
    return {"squared_error": float(aggregate[0]), "share": float(aggregate[0] / aggregate.sum()),
            "by_condition": {key: float(error[0]) for key, error in conditions.items()}}


def _state_summary(state, conditions, family):
    names = REGIME_STATES[family]
    return {"states_lag_scale_and_logit_shift": names,
            "prior_mass": REGIME_PRIORS[family], "mean_mass": state.mean(axis=0).tolist(),
            "mean_mass_by_condition": {condition: state[conditions == condition].mean(axis=0).tolist()
                                       for condition in CONDITIONS}}


def _checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(data, output, *, compare_to=None, extra_rates=()):
    started = perf_counter()
    data, output = Path(data).resolve(), Path(output).resolve()
    if output.exists() or output == data or data in output.parents:
        raise ValidationError("Choose a fresh output directory outside the public data.")
    rates = tuple(dict.fromkeys((48, *extra_rates)))
    metadata = round_metadata(data)
    train_q, train_h = load_inputs(data / "train")
    labels = read_labels(data / "train" / "labels.csv", train_q)
    groups = _conditions(data / "train", train_q)
    cv, folds, n_history, history_support = _cross_validate(train_q, train_h, labels, groups, rates)
    selected = {rate: _choose(cv[rate]) for rate in rates}
    models = {rate: AdaptiveLagForecaster(weight_rate=rate).fit(train_q, train_h, labels) for rate in rates}
    checks = _verify_filter(models[48])
    training_seconds = perf_counter() - started

    queries, history, targets, conditions, reference = tuning_data(data / "validation")
    ids = set(conditions)
    episode_ids = sorted({q.episode_id for q in queries})
    expected_episodes = len(CONDITIONS) * ((metadata["counts_per_condition"]["validation"] + 1) // 2)
    assert len(episode_ids) == expected_episodes and len(ids) == expected_episodes * metadata["machines_per_episode"]
    output.mkdir(parents=True)
    spins, cutoffs = tick_arrays(queries, history)
    machines = np.arange(len(queries))
    group_array = np.array([conditions[q.query_id] for q in queries])
    tuning_support = _support_summary(spins, machines, cutoffs, group_array)
    tuning = {}
    for rate, model in models.items():
        configs, winner = selected[rate]
        start_inference = perf_counter()
        trace = adaptive_trace(queries, spins, model)
        predictions, posteriors = {}, {}
        for name, config in configs.items():
            probability, state = filter_endpoints(trace.base, trace.no_lag_win, spins, machines, cutoffs, config)
            predictions[name] = {q.query_id: tuple(row) for q, row in zip(queries, probability)}
            posteriors[name] = _state_summary(state, group_array, config.family)
        inference_seconds = perf_counter() - start_inference
        checks[f"adaptive_trace_rate_{rate}"] = _verify_trace(queries, history, trace, model, conditions)
        lead = min((name for name in configs if name != "base"),
                   key=lambda name: cv[rate][configs[name]]["cutoffs"]["brier"])
        path = output / ("tuning_predictions.csv" if rate == 48 else f"tuning_predictions_rate_{rate}.csv")
        write_submission(path, predictions[lead])
        predictions[lead] = read_submission(path, ids)
        checks[f"csv_probability_and_id_contract_rate_{rate}"] = True
        if compare_to is not None:
            comparison_path = Path(compare_to) / path.name
            replay = read_submission(comparison_path, ids)
            delta = max(abs(a - b) for key in ids for a, b in zip(replay[key], predictions[lead][key]))
            assert delta <= 1e-8
            checks[f"fresh_replay_rate_{rate}"] = {"comparison_csv": str(comparison_path.resolve()),
                                                  "max_normalized_probability_delta": delta}
        if rate == 48 and data == DEFAULT_PUBLIC:
            matched_control = read_submission(DEFAULT_COMPARISON / "adaptive_lag.csv", ids)
            control_delta = max(abs(a - b) for key in ids for a, b in zip(matched_control[key], predictions["base"][key]))
            assert control_delta <= 1e-12
            checks["saved_adaptive_control_max_abs_delta"] = control_delta
            predictions["base"] = matched_control
        scores = {name: contest_score(pred, targets, conditions, reference, max_points=metadata["max_points"], version=VERSION)
                  for name, pred in predictions.items()}
        if rate == 48 and data == DEFAULT_PUBLIC:
            assert abs(scores["base"]["technical_points"] - 33.224516) < 1e-6
        condition_deltas = []
        for key in ids:
            base = np.asarray(predictions["base"][key][1:])
            for name in predictions:
                values = np.asarray(predictions[name][key][1:])
                condition_deltas.append(float(np.max(np.abs(values / values.sum() - base / base.sum()))))
        max_payout_delta = max(condition_deltas)
        assert max_payout_delta <= 1e-12
        checks[f"conditional_positive_mix_max_abs_delta_rate_{rate}"] = max_payout_delta
        tuning[rate] = {
            "config_by_variant": {name: asdict(config) for name, config in configs.items()},
            "train_cv_winner": asdict(winner), "best_filter_variant_by_train_cv": lead,
            "peer_hazard_status": ("skipped because all local filters lost to the matched base in grouped train"
                                   if winner == BASE else "not part of this local comparison"),
            "tuning_scores": scores, "inference_seconds": inference_seconds,
            "tuning_prediction_csv": path.name, "tuning_prediction_sha256": _checksum(path),
            "p_0_error": {name: _p0_error(pred, targets, conditions) for name, pred in predictions.items()},
            "posterior_mass": posteriors,
            "paired_delta_r_vs_matched_base": {
                name: paired_ratio_interval(pred, predictions["base"], targets, conditions, reference)
                for name, pred in predictions.items() if name != "base"
            },
        }
    report = {
        "round": metadata["round"], "season_id": metadata["season_id"], "split": "tuning",
        "tuning_episodes": episode_ids, "tuning_queries": len(ids), "tuning_window_observation_support": tuning_support,
        "train_cv": {"selection_metric": "episode-mean cutoff five-class one-hot Brier", "fold_seed": 314159,
                     "held_out_episodes": folds, "history_labels": n_history, "cutoff_labels": len(labels),
                     "window_observation_support": history_support,
                     "candidates_by_rate": {rate: [{"config": asdict(config), **value} for config, value in score.items()]
                                            for rate, score in cv.items()}},
        "comparisons_by_adaptive_weight_rate": tuning, "synthetic_seed": SYNTHETIC_SEED,
        "verification": checks, "seconds": {"training_and_verification": training_seconds, "total": perf_counter() - started},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                     "scikit-learn": sklearn.__version__},
        "threads": 2, "command": shlex.join([sys.executable, *sys.argv]),
        "sha256": {"run_regime_filter.py": _checksum(Path(__file__)),
                   "regime_filter.py": _checksum(ROOT / "slot_forecasting" / "regime_filter.py"),
                   "peer_lags.py": _checksum(ROOT / "slot_forecasting" / "peer_lags.py")},
    }
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compare-to", type=Path, help="Prior output directory with tuning CSVs to check.")
    parser.add_argument("--extra-adaptive-rate", type=int, action="append", default=[])
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        report = run(args.data, args.out, compare_to=args.compare_to, extra_rates=args.extra_adaptive_rate)
    for rate, comparison in report["comparisons_by_adaptive_weight_rate"].items():
        name = comparison["best_filter_variant_by_train_cv"]
        score = comparison["tuning_scores"][name]
        print(f"Weight rate {rate}, {name}: {score['technical_points']:.6f} points, R={score['mean_loss_ratio']:.6f}")
    print(args.out / "report.json")


if __name__ == "__main__":
    main()
