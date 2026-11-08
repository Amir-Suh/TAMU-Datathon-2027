"""Train the small self-only TCN from brief 05 and evaluate tuning only."""

import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import platform
import resource
import shlex
import sys
from time import perf_counter

import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_limits
import torch
from torch.nn import functional as F

from run_model_comparison import DEFAULT_COMPARISON, DEFAULT_PUBLIC, _conditions, _outcome_losses, _train_folds, tuning_data
from slot_forecasting.adaptive import AdaptiveLagForecaster
from slot_forecasting.causal import MISSING, tick_arrays
from slot_forecasting.contest import contest_score
from slot_forecasting.io import load_inputs, read_labels, read_submission, write_json, write_submission
from slot_forecasting.peer_lags import adaptive_trace, endpoint_rows
from slot_forecasting.rounds import VERSION, round_metadata
from slot_forecasting.schema import CONDITIONS, PAYOUTS, ValidationError
from slot_forecasting.temporal_cnn import (
    CHANNELS, DILATIONS, KERNEL, WINDOW, CausalTemporalCNN, history_windows, with_win_head,
)
from slot_forecasting.tuning_metrics import paired_ratio_interval


ROOT = Path(__file__).resolve().parent
SEED = 271829
EPOCHS = 4
BATCH_SIZE = 256
LEARNING_RATE = .003
WEIGHT_DECAY = .0001
OPTIMIZED_RATE = 3072


def _fit(features, outcomes, seed):
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 1)
    model = CausalTemporalCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    losses = []
    for _ in range(EPOCHS):
        epoch_loss = 0.
        order = torch.randperm(len(outcomes), generator=generator)
        for rows in order.split(BATCH_SIZE):
            optimizer.zero_grad()
            logits = model(features[rows])
            loss = F.binary_cross_entropy_with_logits(logits, outcomes[rows])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(rows)
        losses.append(epoch_loss / len(outcomes))
    return model.eval(), losses


@torch.inference_mode()
def _predict(model, features):
    return torch.cat([model(batch).sigmoid() for batch in features.split(BATCH_SIZE)]).numpy().astype(float)


def _verify(model):
    rng = np.random.default_rng(SEED)
    spins = rng.integers(MISSING, len(PAYOUTS), size=(4, 90))
    machines = np.arange(len(spins))
    endpoint = 60
    ticks = np.full(len(spins), endpoint)
    spins[0, endpoint - 2:endpoint] = [0, MISSING]
    features = history_windows(spins, machines, ticks)
    np.testing.assert_array_equal(features[0, :, -2], [1, 0, 0, 0, 0, 1])
    np.testing.assert_array_equal(features[0, :, -1], np.zeros(len(PAYOUTS) + 1))

    changed = spins.copy()
    changed[:, endpoint:] = (changed[:, endpoint:] + 1) % len(PAYOUTS)
    altered_windows = history_windows(changed, machines, ticks)
    torch.testing.assert_close(features, altered_windows, rtol=0, atol=0)
    np.testing.assert_array_equal(_predict(model, features), _predict(model, altered_windows))

    with torch.inference_mode():
        altered_features = features.clone()
        altered_features[:, :, 21:] = 0
        torch.testing.assert_close(model.sequence_logits(features)[:, :21],
                                   model.sequence_logits(altered_features)[:, :21], rtol=0, atol=0)
    empty = history_windows(np.full_like(spins, MISSING), machines, ticks)
    assert np.isfinite(_predict(model, empty)).all()
    return {"same_tick_and_future_outcome_invariance": True, "convolution_prefix_invariance": True,
            "missing_distinct_from_zero_payout": True, "finite_all_missing_prefix": True}


def _train(queries, history, labels, conditions):
    spins, cutoffs = tick_arrays(queries, history)
    machines, ticks = endpoint_rows(spins, cutoffs)
    n_history = len(machines) - len(queries)
    outcomes = np.r_[spins[machines[:n_history], ticks[:n_history]],
                     [PAYOUTS.index(labels[q.query_id]) for q in queries]]
    wins = torch.from_numpy((outcomes > 0).astype(np.float32))
    features = history_windows(spins, machines, ticks)
    episodes = np.array([q.episode_id for q in queries])
    endpoint_episodes = episodes[machines]
    oof = {name: np.empty((len(outcomes), len(PAYOUTS))) for name in ("temporal_cnn", "adaptive_48")}
    folds = []
    for index, held_out in enumerate(_train_folds(queries, conditions)):
        held_machines = np.isin(episodes, sorted(held_out))
        held_rows = held_machines[machines]
        fitted, losses = _fit(features[~held_rows], wins[~held_rows], SEED + index + 1)
        fit_q = tuple(q for q in queries if q.episode_id not in held_out)
        fit_h = tuple(p for p in history if p.episode_id not in held_out)
        fit_labels = {q.query_id: labels[q.query_id] for q in fit_q}
        adaptive = AdaptiveLagForecaster().fit(fit_q, fit_h, fit_labels)
        held_q = tuple(q for q in queries if q.episode_id in held_out)
        trace = adaptive_trace(held_q, spins[held_machines], adaptive)
        local_indices = (np.cumsum(held_machines) - 1)[machines[held_rows]]
        base = trace.base[local_indices, ticks[held_rows]]
        oof["adaptive_48"][held_rows] = base
        oof["temporal_cnn"][held_rows] = with_win_head(base, _predict(fitted, features[held_rows]))
        folds.append({"held_out_episodes": sorted(held_out), "seed": SEED + index + 1,
                      "training_binary_log_loss_by_epoch": losses})

    scores = {name: {"historical": _outcome_losses(pred[:n_history], outcomes[:n_history], endpoint_episodes[:n_history]),
                     "cutoffs": _outcome_losses(pred[n_history:], outcomes[n_history:], endpoint_episodes[n_history:])}
              for name, pred in oof.items()}
    fitted, losses = _fit(features, wins, SEED)
    return fitted, {"history_labels": n_history, "cutoff_labels": len(queries), "fold_seed": 314159,
                    "folds": folds, "metrics": scores, "full_training_binary_log_loss_by_epoch": losses}


def _p0_error(prediction, targets, groups):
    errors = {condition: defaultdict(list) for condition in CONDITIONS}
    for t in targets:
        errors[groups[t.query_id]][t.episode_id].append((np.asarray(prediction[t.query_id]) - t.probabilities) ** 2)
    by_condition = {condition: np.mean([np.mean(rows, axis=0) for rows in episodes.values()], axis=0)
                    for condition, episodes in errors.items()}
    mean = np.mean(list(by_condition.values()), axis=0)
    return {"squared_error": float(mean[0]), "share": float(mean[0] / mean.sum()),
            "by_condition": {condition: float(error[0]) for condition, error in by_condition.items()}}


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(data, output, *, compare_to=None):
    begin = perf_counter()
    data, output = Path(data).resolve(), Path(output).resolve()
    if output.exists() or output == data or data in output.parents:
        raise ValidationError("Choose a fresh output directory outside the public data.")
    metadata = round_metadata(data)
    train_q, train_h = load_inputs(data / "train")
    labels = read_labels(data / "train" / "labels.csv", train_q)
    train_conditions = _conditions(data / "train", train_q)
    model, cv = _train(train_q, train_h, labels, train_conditions)
    checks = _verify(model)
    training_seconds = perf_counter() - begin

    queries, history, targets, groups, reference = tuning_data(data / "validation")
    ids = set(groups)
    episodes = sorted({q.episode_id for q in queries})
    expected_episodes = len(CONDITIONS) * ((metadata["counts_per_condition"]["validation"] + 1) // 2)
    assert len(episodes) == expected_episodes and len(ids) == expected_episodes * metadata["machines_per_episode"]
    predictions = {}
    for rate in (48, OPTIMIZED_RATE):
        adaptive = AdaptiveLagForecaster(weight_rate=rate).fit(train_q, train_h, labels)
        predictions[f"adaptive_{rate}"] = adaptive.predict(queries, history)
    if data == DEFAULT_PUBLIC:
        saved = read_submission(DEFAULT_COMPARISON / "adaptive_lag.csv", ids)
        delta = max(abs(a - b) for key in ids for a, b in zip(saved[key], predictions["adaptive_48"][key]))
        assert delta <= 1e-12
        checks["adaptive_48_saved_control_max_delta"] = delta
        predictions["adaptive_48"] = saved

    base = np.asarray([predictions["adaptive_48"][q.query_id] for q in queries])
    start_inference = perf_counter()
    spins, cutoffs = tick_arrays(queries, history)
    features = history_windows(spins, np.arange(len(queries)), cutoffs)
    forecast = with_win_head(base, _predict(model, features))
    inference_seconds = perf_counter() - start_inference
    candidate_sizes = forecast[:, 1:] / forecast[:, 1:].sum(axis=1, keepdims=True)
    checks["adaptive_48_positive_mix_max_delta"] = float(np.max(np.abs(
        candidate_sizes - base[:, 1:] / base[:, 1:].sum(axis=1, keepdims=True))))
    assert checks["adaptive_48_positive_mix_max_delta"] <= 1e-12
    output.mkdir(parents=True)
    checkpoint = output / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    path = output / "tuning_predictions.csv"
    write_submission(path, {q.query_id: tuple(row) for q, row in zip(queries, forecast)})
    predictions["temporal_cnn"] = read_submission(path, ids)
    checks["probability_and_query_id_contract"] = True
    if compare_to is not None:
        prior_run = read_submission(compare_to, ids)
        delta = max(abs(a - b) for key in ids for a, b in zip(prior_run[key], predictions["temporal_cnn"][key]))
        assert delta <= 1e-8
        checks["fresh_replay"] = {"comparison_csv": str(Path(compare_to).resolve()),
                                  "max_normalized_probability_delta": delta}

    scores = {name: contest_score(prediction, targets, groups, reference, max_points=metadata["max_points"], version=VERSION)
              for name, prediction in predictions.items()}
    if data == DEFAULT_PUBLIC:
        assert abs(scores["adaptive_48"]["technical_points"] - 33.224516) < 1e-6
        assert abs(scores["adaptive_3072"]["technical_points"] - 44.097892) < 1e-6
    peak_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** (2 if sys.platform == "darwin" else 1))
    report = {
        "round": metadata["round"], "season_id": metadata["season_id"], "split": "tuning",
        "tuning_episodes": episodes, "tuning_queries": len(ids),
        "config": {"window": WINDOW, "channels": CHANNELS, "kernel": KERNEL, "dilations": DILATIONS,
                   "receptive_field": 1 + (KERNEL - 1) * sum(DILATIONS), "last_state_pooling": True,
                   "optimizer": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
                   "epochs": EPOCHS, "batch_size": BATCH_SIZE, "full_fit_seed": SEED},
        "train_cv": cv, "tuning_scores": scores, "verification": checks,
        "p_0_error": {name: _p0_error(prediction, targets, groups) for name, prediction in predictions.items()},
        "paired_delta_r_vs_adaptive": {str(rate): paired_ratio_interval(predictions["temporal_cnn"],
                                                                    predictions[f"adaptive_{rate}"], targets, groups, reference)
                                      for rate in (48, OPTIMIZED_RATE)},
        "seconds": {"training_and_verification": training_seconds, "tuning_candidate_inference": inference_seconds,
                    "total": perf_counter() - begin}, "peak_process_memory_mib": peak_memory,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                     "scikit-learn": sklearn.__version__, "scipy": scipy.__version__},
        "cpu_threads": {"intraop": torch.get_num_threads(), "interop": torch.get_num_interop_threads()},
        "command": shlex.join([sys.executable, *sys.argv]),
        "sha256": {"tuning_predictions.csv": _sha256(path), "model.pt": _sha256(checkpoint),
                   "run_temporal_cnn.py": _sha256(Path(__file__)),
                   "temporal_cnn.py": _sha256(ROOT / "slot_forecasting" / "temporal_cnn.py")},
    }
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compare-to", type=Path, help="Prior run's tuning CSV for the deterministic check.")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=2):
        report = run(args.data, args.out, compare_to=args.compare_to)
    score = report["tuning_scores"]["temporal_cnn"]
    print(f"Temporal CNN: {score['technical_points']:.6f} points, R={score['mean_loss_ratio']:.6f}")
    print(args.out / "report.json")


if __name__ == "__main__":
    main()
