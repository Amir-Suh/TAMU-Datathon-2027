"""Freeze and score the main-round audit comparison in separate steps."""

import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from run_bayesian_lag_pairs import LEAD_CSV, LEAD_SHA256
from run_boosted_correction import combine_samples, endpoint_sample
from run_model_comparison import DEFAULT_PUBLIC, ROOT, _conditions, _train_folds
from slot_forecasting.adaptive import AdaptiveLagForecaster
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.bayesian_lag_pairs import BayesianLagPairForecaster, LagPairConfig
from slot_forecasting.boosted_correction import OffsetTrees, build_inputs
from slot_forecasting.causal import tick_arrays
from slot_forecasting.combine_forecasts import conditional_positive, load_component, with_positive_head
from slot_forecasting.contest import contest_score
from slot_forecasting.experiments import episode_folds
from slot_forecasting.io import load_inputs, read_history, read_labels, read_queries, read_submission, read_targets, require_ids, write_json, write_submission
from slot_forecasting.peer_lags import adjust_win
from slot_forecasting.rounds import VERSION, round_metadata
from slot_forecasting.schema import CONDITIONS, ValidationError
from slot_forecasting.tuning_metrics import paired_ratio_interval


DATA = DEFAULT_PUBLIC
BAYES_DIR = ROOT / "runs/bayesian-lag-pairs/replay"
BOOST_CONFIG = ROOT / "runs/04-boosted-correction/main-rate3072/config.json"
LOGISTIC = ROOT / "models/logistic_regression/two_stage_self.joblib"
LOGISTIC_TUNING = ROOT / "runs/model-comparison/two_stage_self.csv"
BAYES_CONFIG = LagPairConfig(evidence_half_life=None, hazard=1/96)
N_TREES = 60
N_THREADS = 2


def _json(path):
    return json.loads(path.read_bytes())


def _digest(path):
    return sha256(path.read_bytes()).hexdigest()


def _audit_manifest():
    validation = DATA / "validation"
    queries = read_queries(validation / "queries.csv")
    groups = _conditions(validation, queries)
    tuning, audit = episode_folds(queries, groups)
    if set(_json(BAYES_DIR / "report.json")["tuning_episodes"]) != tuning:
        raise ValidationError("The validation partition differs from the recorded tuning partition.")
    audit_queries = tuple(q for q in queries if q.episode_id in audit)
    groups = {q.query_id: groups[q.query_id] for q in audit_queries}
    episode_groups = {q.episode_id: groups[q.query_id] for q in audit_queries}
    by_condition = Counter(episode_groups.values())
    if len(audit) != 48 or len(audit_queries) != 384 or by_condition != {c: 12 for c in CONDITIONS}:
        raise ValidationError("Expected 48 audit episodes, 12 per condition, and 384 audit queries.")
    tuning_queries = tuple(q for q in queries if q.episode_id in tuning)
    return tuning, audit, tuning_queries, audit_queries, groups


def _fit_frozen_boost():
    config = _json(BOOST_CONFIG)
    if config["lead_correction"] != f"trees_{N_TREES}" or config["adaptive"]["weight_rate"] != 3072:
        raise ValidationError("The original lead used 60 trees and adaptive rate 3072.")
    train = DATA / "train"
    queries, history = load_inputs(train)
    labels = read_labels(train / "labels.csv", queries)
    groups = _conditions(train, queries)
    samples, inputs = [], []
    # Rebuild the original final fit's out-of-episode features. There is no model selection.
    for holdout in _train_folds(queries, groups):
        held_queries = tuple(q for q in queries if q.episode_id in holdout)
        held_history = (p for p in history if p.episode_id in holdout)
        fit_queries = tuple(q for q in queries if q.episode_id not in holdout)
        fit_history = (p for p in history if p.episode_id not in holdout)
        fit_labels = {q.query_id: labels[q.query_id] for q in fit_queries}
        base = AdaptiveLagForecaster(**config["adaptive"]).fit(fit_queries, fit_history, fit_labels)
        sample = endpoint_sample(held_queries, held_history, labels)
        samples.append(sample)
        inputs.append(sample.adaptive_inputs(base))

    features, outcomes, weights = combine_samples(samples, inputs)
    tree_keys = ("max_depth", "min_samples_leaf", "shrinkage", "step_limit", "curvature_floor", "seed")
    settings = {key: config[key] for key in tree_keys}
    trees = OffsetTrees(rounds=N_TREES, **settings).fit(features.features, features.offset, outcomes, weights)
    adaptive = AdaptiveLagForecaster(**config["adaptive"]).fit(queries, history, labels)
    return adaptive, trees, {"adaptive": config["adaptive"], "trees": {"rounds": N_TREES, **settings}}


def _predict(queries, history, adaptive, trees, bayesian, logistic):
    ids = [q.query_id for q in queries]
    logistic_prediction = logistic.predict(queries, history)
    logistic_array = np.array([logistic_prediction[key] for key in ids])
    prior = np.asarray(bayesian.prior.global_prior)[1:]
    positive = conditional_positive(logistic_array, prior / prior.sum())
    bayesian_prediction = bayesian.predict(queries, history)
    spins, cutoffs = tick_arrays(queries, history)
    features = build_inputs(queries, spins, np.arange(len(queries)), cutoffs, adaptive)
    boosted = adjust_win(features.base, trees.corrections(features.features, (N_TREES,))[N_TREES])
    return {
        "bayesian": with_positive_head(np.array([bayesian_prediction[key][0] for key in ids]), positive),
        "previous_lead": with_positive_head(boosted[:, 0], positive),
        "logistic": logistic_array,
    }


def prepare(output):
    if output.exists():
        raise FileExistsError(output)
    tuning, audit, tuning_queries, audit_queries, _ = _audit_manifest()
    if _json(BAYES_DIR / "report.json")["config"] != asdict(BAYES_CONFIG):
        raise ValidationError("Bayesian freeze must match the training-selected 1/96 configuration.")
    if _digest(LEAD_CSV) != LEAD_SHA256:
        raise ValidationError("The saved 50.274 tuning forecast has changed.")
    bayesian = BayesianLagPairForecaster(BAYES_CONFIG)
    bayesian.prior = FrequencyForecaster.load(BAYES_DIR / "population_prior.json")
    logistic = joblib.load(LOGISTIC)
    adaptive, trees, boost = _fit_frozen_boost()

    # Forecast-only replay against saved tuning CSVs verifies the exact prior lead and heads.
    validation = DATA / "validation"
    tuning_h = read_history(validation / "history.csv", episode_ids=tuning)
    replay = _predict(tuning_queries, tuning_h, adaptive, trees, bayesian, logistic)
    tuning_ids = [q.query_id for q in tuning_queries]
    references = {"bayesian": BAYES_DIR / "standalone.csv", "previous_lead": LEAD_CSV,
                  "logistic": LOGISTIC_TUNING}
    deltas = {key: float(np.max(np.abs(replay[key] - load_component(path, tuning_ids)[0])))
              for key, path in references.items()}
    if max(deltas.values()) > 1e-12:
        raise ValidationError(f"Frozen forecasts failed the saved tuning replay: {deltas}")

    output.mkdir(parents=True)
    joblib.dump({"bayesian": bayesian, "adaptive": adaptive, "trees": trees, "logistic": logistic},
                output / "frozen_models.joblib")
    audit_h = read_history(validation / "history.csv", episode_ids=audit)
    forecast = _predict(audit_queries, audit_h, adaptive, trees, bayesian, logistic)
    hashes = {}
    audit_ids = [q.query_id for q in audit_queries]
    for name in ("bayesian", "previous_lead"):
        path = output / f"{name}.csv"
        write_submission(path, dict(zip(audit_ids, forecast[name])))
        read_submission(path, audit_ids)
        hashes[name] = _digest(path)
    write_json(output / "freeze.json", {
        "split": "audit", "queries": len(audit_ids), "episode_ids": sorted(audit),
        "query_episodes": {q.query_id: q.episode_id for q in audit_queries},
        "bayesian": asdict(BAYES_CONFIG), "previous_lead": boost,
        "positive_payout_head": {"source": str(LOGISTIC.relative_to(ROOT)), "sha256": _digest(LOGISTIC)},
        "training_source": "Original train split only. Bayesian prior and logistic checkpoint loaded from saved models.",
        "model_sha256": _digest(output / "frozen_models.joblib"), "prediction_sha256": hashes,
        "tuning_replay_max_abs_delta": deltas,
    })
    print(f"Frozen 2 models and their 384 audit forecasts in {output}. Tuning replay deltas: {deltas}")


def score(output):
    freeze = _json(output / "freeze.json")
    if (output / "report.json").exists():
        raise FileExistsError(output / "report.json")
    if _digest(output / "frozen_models.joblib") != freeze["model_sha256"]:
        raise ValidationError("Frozen model parameters have changed.")
    _, audit, _, audit_queries, groups = _audit_manifest()
    ids = set(groups)
    expected_episodes = {q.query_id: q.episode_id for q in audit_queries}
    if freeze["query_episodes"] != expected_episodes or freeze["episode_ids"] != sorted(audit):
        raise ValidationError("Audit episode IDs differ from the frozen manifest.")
    predictions = {}
    for name, digest in freeze["prediction_sha256"].items():
        path = output / f"{name}.csv"
        if _digest(path) != digest:
            raise ValidationError(f"Frozen forecast changed: {path}")
        predictions[name] = read_submission(path, ids)

    # This is the first step that loads audit answer or reference values.
    validation = DATA / "validation"
    targets = read_targets(validation / "targets.csv", query_ids=ids)
    require_ids({target.query_id for target in targets}, ids)
    if any(expected_episodes[t.query_id] != t.episode_id for t in targets):
        raise ValidationError("Audit targets do not match the frozen query manifest.")
    reference = read_submission(validation / "reference.csv", ids, selection=("query_id", ids))
    max_points = round_metadata(DATA)["max_points"]
    scores = {name: contest_score(prediction, targets, groups, reference, max_points=max_points, version=VERSION)
              for name, prediction in predictions.items()}
    interval = paired_ratio_interval(predictions["bayesian"], predictions["previous_lead"], targets, groups, reference)
    report = {
        "split": "audit", "episodes": len(audit), "queries": len(ids),
        "scores": scores, "points_delta": scores["bayesian"]["technical_points"] - scores["previous_lead"]["technical_points"],
        "paired_delta_r": interval,
        "freeze_sha256": _digest(output / "freeze.json"),
        "prediction_sha256": freeze["prediction_sha256"],
    }
    write_json(output / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "score"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=N_THREADS):
        (prepare if args.phase == "prepare" else score)(args.out)
