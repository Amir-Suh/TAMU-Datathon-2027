"""Select signed motifs on grouped training folds, then score tuning only."""

import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import platform
import shlex
import sys
from time import perf_counter

import numpy as np
import scipy
from scipy.special import expit, logit
import sklearn
from threadpoolctl import threadpool_limits

from run_model_comparison import DEFAULT_COMPARISON, DEFAULT_PUBLIC, _conditions, _outcome_losses, _train_folds, tuning_data
from slot_forecasting.adaptive import AdaptiveLagForecaster
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.causal import MISSING, tick_arrays
from slot_forecasting.contest import contest_score
from slot_forecasting.io import load_inputs, read_labels, read_submission, write_json, write_submission
from slot_forecasting.rounds import VERSION, round_metadata
from slot_forecasting.schema import CONDITIONS, PAYOUTS, ValidationError
from slot_forecasting.signed_motifs import (
    AMPLITUDES, RECENCY_DECAY, MotifConfig, SignedMotifForecaster,
    hypothesis_prior, motif_rollout,
)
from slot_forecasting.tuning_metrics import paired_ratio_interval


ROOT = Path(__file__).resolve().parent
SYNTHETIC_SEED = 271828
HISTORY_ENDPOINTS = 32
GRID = (MotifConfig("null"), *(
    MotifConfig(family, half_life)
    for family in ("singleton", "singleton_pair") for half_life in (None, 48)
))


def _verify_rollout():
    rng = np.random.default_rng(SYNTHETIC_SEED)
    config = MotifConfig()
    prior = np.array([.55, .31, .09, .04, .01])
    priors = np.tile(prior, (3, 1))
    n_observations = 1400
    spins = np.full((3, n_observations + 1), MISSING, dtype=np.int8)
    wins, totals = np.zeros(3), np.zeros(3)
    effects = np.array([2, -2, 0])
    for tick in range(n_observations):
        base = (8 * (1 - prior[0]) + wins) / (8 + totals)
        if tick >= 5:
            sign = (2 * spins[:, tick - 2] - 1) * (2 * spins[:, tick - 5] - 1)
            base = expit(logit(base) + effects * sign)
        spins[:, tick] = rng.random(3) < base
        wins = RECENCY_DECAY * wins + spins[:, tick]
        totals = RECENCY_DECAY * totals + 1

    forecast, details = motif_rollout(spins, priors, config,
                                     diagnostic_ticks=np.full(3, n_observations))
    null = motif_rollout(spins, priors, MotifConfig("null"))[0]
    midpoint = n_observations // 2
    active = details["motifs"].index((2, 5)) * len(AMPLITUDES) + 1
    evaluation = []
    for row, effect in enumerate(effects):
        posterior = details["posterior"][row]
        hypothesis = active + AMPLITUDES.index(effect) if effect else 0
        assert posterior.argmax() == hypothesis and posterior[hypothesis] > .9
        labels = spins[row, midpoint:n_observations]
        brier = float(np.mean((1 - forecast[row, midpoint:n_observations, 0] - labels) ** 2))
        null_brier = float(np.mean((1 - null[row, midpoint:n_observations, 0] - labels) ** 2))
        if effect:
            assert brier < null_brier
        evaluation.append({"true_amplitude": int(effect), "true_hypothesis_mass": float(posterior[hypothesis]),
                           "held_out_win_brier": brier, "held_out_null_win_brier": null_brier})

    synthetic = spins[:, :60].copy()
    changed = synthetic.copy()
    changed[:, 35:] = 1 - changed[:, 35:]
    checks = {}
    for candidate in GRID:
        original = motif_rollout(synthetic, priors, candidate)[0]
        alternative = motif_rollout(changed, priors, candidate)[0]
        np.testing.assert_array_equal(original[:, :36], alternative[:, :36])
        missing = np.full_like(synthetic, MISSING)
        pred, diagnostic = motif_rollout(missing, priors, candidate, diagnostic_ticks=np.full(3, 59))
        np.testing.assert_allclose(pred, np.broadcast_to(prior, pred.shape), atol=1e-14, rtol=0)
        state = hypothesis_prior(candidate)
        np.testing.assert_allclose(diagnostic["posterior"], np.tile(state, (3, 1)), atol=1e-14, rtol=0)
        assert not diagnostic["observed_support"].any()
    checks.update(synthetic_effects=evaluation, same_tick_and_future_invariance=True,
                  all_missing_forecast_and_evidence=True)
    return checks


def _cross_validate(queries, history, labels, groups):
    spins, cutoffs = tick_arrays(queries, history)
    episodes = np.array([query.episode_id for query in queries])
    outcomes = np.array([PAYOUTS.index(labels[q.query_id]) for q in queries])
    rows, ticks = zip(*[(row, tick) for row, spin in enumerate(spins)
                        for tick in np.flatnonzero(spin != MISSING)[-HISTORY_ENDPOINTS:]])
    rows, ticks = np.asarray(rows), np.asarray(ticks)
    histories = {config: np.empty((len(rows), len(PAYOUTS))) for config in GRID}
    actual = {config: np.empty((len(queries), len(PAYOUTS))) for config in GRID}
    folds = []
    for held_out in _train_folds(queries, groups):
        folds.append(sorted(held_out))
        fit_queries = tuple(q for q in queries if q.episode_id not in held_out)
        fit_history = tuple(p for p in history if p.episode_id not in held_out)
        fitted = SignedMotifForecaster().fit(fit_queries, fit_history,
                                            {q.query_id: labels[q.query_id] for q in fit_queries})
        mask = np.isin(episodes, sorted(held_out))
        select_history = mask[rows]
        machine_rows = np.flatnonzero(mask)
        local_rows = np.cumsum(mask) - 1
        priors = fitted.prior_rows(tuple(queries[row] for row in machine_rows))
        for config in GRID:
            prediction = motif_rollout(spins[mask], priors, config)[0]
            histories[config][select_history] = prediction[local_rows[rows[select_history]], ticks[select_history]]
            actual[config][mask] = prediction[np.arange(len(machine_rows)), cutoffs[mask]]
    scores = {
        config: {"historical": _outcome_losses(histories[config], spins[rows, ticks], episodes[rows]),
                 "cutoffs": _outcome_losses(actual[config], outcomes, episodes)}
        for config in GRID
    }
    return scores, folds, len(rows)


def _episode_errors(predictions, targets, groups):
    errors = {}
    for condition in CONDITIONS:
        episodes = sorted({t.episode_id for t in targets if groups[t.query_id] == condition})
        errors[condition] = {
            episode: np.mean([(np.asarray(predictions[t.query_id]) - t.probabilities) ** 2
                              for t in targets if t.episode_id == episode], axis=0)
            for episode in episodes
        }
    return errors


def _p0_error(episode_errors):
    by_condition = {name: np.mean(list(episodes.values()), axis=0) for name, episodes in episode_errors.items()}
    raw = np.mean(list(by_condition.values()), axis=0)
    return {"squared_error": float(raw[0]), "share": float(raw[0] / raw.sum()),
            "by_condition": {name: float(value[0]) for name, value in by_condition.items()}}


def _leave_best_out(candidate, control, reference):
    result = {}
    for condition in CONDITIONS:
        deltas = {key: float((error - control[condition][key]).sum())
                  for key, error in candidate[condition].items()}
        if len(deltas) < 2:
            continue
        best = min(deltas, key=deltas.get)
        kept = sorted(set(deltas) - {best})
        denom = max(1e-9, float(np.mean([reference[condition][key].sum() for key in kept])))
        result[condition] = {"dropped_episode": best, "dropped_raw_error_delta": deltas[best],
                             "delta_r": float(np.mean([deltas[key] for key in kept]) / denom)}
    return result


def _quantiles(values):
    return dict(zip(("p10", "p50", "p90"), map(float, np.quantile(values, [.1, .5, .9]))))


def _posterior_summary(details):
    posterior = details["posterior"]
    winners = posterior.argmax(axis=1)
    rows = []
    for i, motif in enumerate(details["motifs"]):
        start = i * len(AMPLITUDES) + 1
        active = (winners >= start) & (winners < start + len(AMPLITUDES))
        support = details["observed_support"][:, i]
        available = details["available_at_cutoff"][:, i]
        rows.append({"lags": motif, "mean_posterior_mass": float(posterior[:, start:start + len(AMPLITUDES)].sum(axis=1).mean()),
                     "queries_where_top_hypothesis": int(active.sum()),
                     "top_and_available_queries": int((active & available).sum()),
                     "observed_history_support": _quantiles(support),
                     "support_when_top_and_available": _quantiles(support[active & available]) if (active & available).any() else None})
    return {"max_hypothesis_mass": _quantiles(posterior.max(axis=1)),
            "null_mass": _quantiles(posterior[:, 0]), "motifs": rows}


def _checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(data, output, controls=None, compare_to=None):
    started = perf_counter()
    data, output = Path(data).resolve(), Path(output).resolve()
    if output.exists() or output == data or data in output.parents:
        raise ValidationError("Choose a fresh output directory outside the public data.")
    metadata = round_metadata(data)
    checks = _verify_rollout()
    queries, history = load_inputs(data / "train")
    labels = read_labels(data / "train" / "labels.csv", queries)
    groups = _conditions(data / "train", queries)
    cv, folds, history_rows = _cross_validate(queries, history, labels, groups)
    selected = min(GRID, key=lambda config: cv[config]["cutoffs"]["brier"])
    family_best = {family: min((c for c in GRID if c.family == family), key=lambda c: cv[c]["cutoffs"]["brier"])
                   for family in ("null", "singleton", "singleton_pair")}
    fitted = SignedMotifForecaster(selected).fit(queries, history, labels)
    training_seconds = perf_counter() - started

    tuning_q, tuning_h, targets, conditions, reference = tuning_data(data / "validation")
    ids = set(conditions)
    episode_ids = sorted({q.episode_id for q in tuning_q})
    expected_episodes = len(CONDITIONS) * ((metadata["counts_per_condition"]["validation"] + 1) // 2)
    assert len(episode_ids) == expected_episodes and len(ids) == expected_episodes * metadata["machines_per_episode"]
    spins, cutoffs = tick_arrays(tuning_q, tuning_h)
    priors = fitted.prior_rows(tuning_q)
    output.mkdir(parents=True)
    predictions, posterior = {}, {}
    inference_start = perf_counter()
    for name, config in family_best.items():
        forecast, details = motif_rollout(spins, priors, config, diagnostic_ticks=cutoffs)
        predictions[name] = {q.query_id: tuple(row) for q, row in zip(tuning_q, forecast[np.arange(len(tuning_q)), cutoffs])}
        posterior[name] = _posterior_summary(details)
    inference_seconds = perf_counter() - inference_start
    baseline = FrequencyForecaster("pooled_recent", prior_strength=8, half_life=12).fit(queries, history, labels)
    recency = baseline.predict(tuning_q, tuning_h)
    null_delta = max(abs(predictions["null"][key][0] - recency[key][0]) for key in ids)
    assert null_delta <= 1e-12
    checks["null_vs_recency_win_max_abs_delta"] = null_delta

    if controls is None and data == DEFAULT_PUBLIC:
        controls = DEFAULT_COMPARISON
    if controls is None:
        predictions.update(AdaptiveLagForecaster().fit(queries, history, labels).predict_variants(tuning_q, tuning_h))
    else:
        controls = Path(controls).resolve()
        for name in ("adaptive_lag", "lag_fixed"):
            predictions[name] = read_submission(controls / f"{name}.csv", ids)
    conditional_deltas = []
    for key in ids:
        adaptive_sizes = np.asarray(predictions["adaptive_lag"][key][1:])
        for family in family_best:
            sizes = np.asarray(predictions[family][key][1:])
            conditional_deltas.append(float(np.max(np.abs(sizes / sizes.sum() - adaptive_sizes / adaptive_sizes.sum()))))
    checks["conditional_positive_mix_max_abs_delta"] = max(conditional_deltas)
    assert checks["conditional_positive_mix_max_abs_delta"] <= 1e-12

    path = output / "tuning_predictions.csv"
    write_submission(path, predictions[selected.family])
    predictions[selected.family] = read_submission(path, ids)
    checks["csv_probability_and_id_contract"] = True
    if compare_to is not None:
        replay = read_submission(compare_to, ids)
        replay_delta = max(abs(a - b) for key in ids for a, b in zip(replay[key], predictions[selected.family][key]))
        assert replay_delta <= 1e-8
        checks["fresh_replay"] = {"comparison_csv": str(Path(compare_to).resolve()),
                                  "max_normalized_probability_delta": replay_delta}

    scores = {name: contest_score(pred, targets, conditions, reference, max_points=metadata["max_points"], version=VERSION)
              for name, pred in predictions.items()}
    if data == DEFAULT_PUBLIC:
        assert abs(scores["adaptive_lag"]["technical_points"] - 33.224516) < 1e-6
    errors = {name: _episode_errors(pred, targets, conditions) for name, pred in predictions.items()}
    reference_errors = _episode_errors(reference, targets, conditions)
    report = {
        "round": metadata["round"], "season_id": metadata["season_id"], "split": "tuning",
        "tuning_episodes": episode_ids, "tuning_queries": len(ids),
        "config": asdict(selected), "amplitudes": AMPLITUDES,
        "prior_family_mass": {"null": [1, 0, 0], "singleton": [.5, .5, 0], "singleton_pair": [.5, .25, .25]},
        "train_cv": {"selection_metric": "episode-mean cutoff five-class one-hot Brier", "fold_seed": 314159,
                     "held_out_episodes": folds, "history_labels": history_rows, "cutoff_labels": len(labels),
                     "candidates": [{"config": asdict(config), **value} for config, value in cv.items()]},
        "best_config_by_family": {name: asdict(config) for name, config in family_best.items()},
        "scores": scores, "p_0_error": {name: _p0_error(error) for name, error in errors.items()},
        "paired_delta_r_vs_adaptive": paired_ratio_interval(predictions[selected.family], predictions["adaptive_lag"],
                                                             targets, conditions, reference),
        "leave_largest_gain_out_vs_adaptive": _leave_best_out(errors[selected.family], errors["adaptive_lag"], reference_errors),
        "posterior_by_family": posterior, "verification": checks, "synthetic_seed": SYNTHETIC_SEED,
        "seconds": {"training_and_verification": training_seconds, "tuning_inference_all_families": inference_seconds,
                    "total": perf_counter() - started},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                     "scikit-learn": sklearn.__version__},
        "threads": 2, "command": shlex.join([sys.executable, *sys.argv]),
        "sha256": {"tuning_predictions.csv": _checksum(path), "run_signed_motifs.py": _checksum(Path(__file__)),
                   "signed_motifs.py": _checksum(ROOT / "slot_forecasting" / "signed_motifs.py")},
    }
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--controls", type=Path)
    parser.add_argument("--compare-to", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        report = run(args.data, args.out, args.controls, args.compare_to)
    name = report["config"]["family"]
    score = report["scores"][name]
    print(f"{name}: {score['technical_points']:.6f} points, R={score['mean_loss_ratio']:.6f}")
    print(args.out / "report.json")


if __name__ == "__main__":
    main()
