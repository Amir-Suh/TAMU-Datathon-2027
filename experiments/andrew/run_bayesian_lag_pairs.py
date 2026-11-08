"""Select Bayesian lag-pair variants on grouped train folds and score tuning only."""

import argparse
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
import platform
import shlex
import sys
from time import perf_counter

import numpy as np
import scipy
from threadpoolctl import threadpool_limits

from run_model_comparison import (
    DEFAULT_COMPARISON, DEFAULT_PUBLIC, ROOT, _conditions, _outcome_losses, _train_folds, tuning_data,
)
from slot_forecasting.bayesian_lag_pairs import BayesianLagPairForecaster, LagPairConfig, lag_pair_rollout
from slot_forecasting.causal import LAG_PAIRS, tick_arrays
from slot_forecasting.combine_forecasts import (
    conditional_positive, convex_mix, crossfit_blend, load_component, with_positive_head,
)
from slot_forecasting.contest import contest_score
from slot_forecasting.io import load_inputs, read_labels, write_json, write_submission
from slot_forecasting.peer_lags import endpoint_rows
from slot_forecasting.rounds import VERSION, round_metadata
from slot_forecasting.schema import CONDITIONS, PAYOUTS, ValidationError
from slot_forecasting.scoring import score_predictions
from slot_forecasting.signed_motifs import MotifConfig, motif_rollout
from slot_forecasting.tuning_metrics import paired_ratio_interval

LEAD_CSV = ROOT / "runs/06-combine-forecasts/replay/fixed__boosted_3072_logistic_head.csv"
SIGNED_CSV = ROOT / "runs/01-signed-motifs/replay/tuning_predictions.csv"
LEAD_SHA256 = "0530eb2bbd4e3738c7df88c3412e5501a65da7e21f23355e0db9585c4d1d6fd5"
LEAD_POINTS = 50.27395719034001
LEAD_RATIO = .19346085634032428
SIGNED_POINTS = 42.401934374422645
BLEND_WEIGHT = .25
GRID = {
    "signed_48": LagPairConfig(),
    "wide_signed_48": LagPairConfig(wide=True),
    "signed_static": LagPairConfig(evidence_half_life=None),
    "switch_96": LagPairConfig(evidence_half_life=None, hazard=1/96),
    "switch_32": LagPairConfig(evidence_half_life=None, hazard=1/32),
    "parity_table": LagPairConfig(family="parity_table", evidence_half_life=None),
    "four_cell_table": LagPairConfig(family="four_cell_table", evidence_half_life=None),
}


def _win_losses(no_payout, outcomes, episodes):
    result = _outcome_losses(np.column_stack((no_payout, 1 - no_payout)), (outcomes > 0).astype(int), episodes)
    result["brier"] *= .5
    return result


def _train_validation(queries, history, labels, groups):
    spins, cutoffs = tick_arrays(queries, history)
    machines, ticks = endpoint_rows(spins, cutoffs)
    cutoff = ticks == cutoffs[machines]
    outcomes = spins[machines, ticks].copy()
    outcomes[cutoff] = [PAYOUTS.index(labels[q.query_id]) for q in queries]
    episodes = np.array([q.episode_id for q in queries])
    endpoint_episodes = episodes[machines]
    endpoint_conditions = np.array([groups[q.query_id] for q in queries])[machines]
    folds = list(_train_folds(queries, groups))
    fold_data = []
    for holdout in folds:
        fit_queries = tuple(q for q in queries if q.episode_id not in holdout)
        fit_history = (play for play in history if play.episode_id not in holdout)
        fit_labels = {q.query_id: labels[q.query_id] for q in fit_queries}
        mask = np.isin(episodes, sorted(holdout))
        endpoint_mask = mask[machines]
        fit = BayesianLagPairForecaster().fit(fit_queries, fit_history, fit_labels)
        priors = fit.prior_rows(tuple(q for q, held in zip(queries, mask) if held))
        relative_machines = (np.cumsum(mask) - 1)[machines[endpoint_mask]]
        fold_data.append((spins[mask], priors, endpoint_mask, relative_machines, ticks[endpoint_mask]))

    def evaluate(config):
        no_payout = np.empty(len(ticks))
        for spin, priors, mask, rows, selected_ticks in fold_data:
            forecast, _ = lag_pair_rollout(spin, priors, config)
            no_payout[mask] = forecast[rows, selected_ticks, 0]
        return {
            "config": asdict(config),
            "cutoffs": _win_losses(no_payout[cutoff], outcomes[cutoff], endpoint_episodes[cutoff]),
            "historical": _win_losses(no_payout[~cutoff], outcomes[~cutoff], endpoint_episodes[~cutoff]),
            "cutoffs_by_condition": {
                condition: _win_losses(no_payout[subset], outcomes[subset], endpoint_episodes[subset])
                for condition in CONDITIONS
                for subset in (cutoff & (endpoint_conditions == condition),)
            },
        }

    grid = GRID.copy()
    rows = {name: evaluate(config) for name, config in grid.items()}

    def brier(name):
        return rows[name]["cutoffs"]["brier"]

    best_switch = min(("switch_96", "switch_32"), key=brier)
    if brier("wide_signed_48") < brier("signed_48") and brier(best_switch) < brier("signed_static"):
        name = "wide_" + best_switch
        grid[name] = replace(grid[best_switch], wide=True)
        rows[name] = evaluate(grid[name])
    winner = min(grid, key=brier)
    return winner, grid[winner], {
        "selection": "Minimum episode-weighted binary Brier on held-out training cutoffs.",
        "selected": winner,
        "cutoff_labels": int(cutoff.sum()),
        "history_labels": int((~cutoff).sum()),
        "held_out_episodes": [sorted(fold) for fold in folds],
        "candidates": rows,
    }


def _posterior_summary(diagnostics, conditions):
    results = {}
    for condition in CONDITIONS:
        subset = conditions == condition
        pair = diagnostics["pair_posterior"][subset]
        mass = pair.sum(axis=1)
        conditional = pair / np.maximum(mass[:, None], 1e-300)
        entropy = -(conditional * np.log(np.maximum(conditional, 1e-300))).sum(axis=1)
        support = diagnostics["pair_support"][subset]
        top = pair.argmax(axis=1)
        counts = Counter(LAG_PAIRS[i] for i in top)
        row = {
            "mean_pair_mass": float(mass.mean()),
            "median_pair_entropy": float(np.median(entropy)),
            "median_usable_observations_for_top_pair": float(np.median(support[np.arange(len(top)), top])),
            "top_pairs": [{"lags": pair, "queries": count} for pair, count in counts.most_common()],
            "available_pair_fraction": float(diagnostics["available_at_cutoff"][subset].mean()),
        }
        if "amplitudes" in diagnostics:
            amplitudes = diagnostics["amplitudes"]
            amplitude_posterior = diagnostics["amplitude_posterior"][subset]
            total = max(float(mass.sum()), 1e-300)
            row["expected_signed_amplitude"] = float((amplitude_posterior @ amplitudes).sum() / total)
            row["edge_mass_given_pair"] = float(
                amplitude_posterior[:, abs(amplitudes) == max(abs(amplitudes))].sum() / total)
        results[condition] = row
    return results


def _probability_diagnostics(predictions, targets, groups):
    squared = {condition: {} for condition in CONDITIONS}
    for row in targets:
        error = np.square(np.asarray(predictions[row.query_id]) - row.probabilities)
        squared[groups[row.query_id]].setdefault(row.episode_id, []).append(error)
    by_condition = {
        condition: np.mean([np.mean(errors, axis=0) for errors in episodes.values()], axis=0)
        for condition, episodes in squared.items()
    }
    calibration = score_predictions(predictions, targets)
    return {
        "raw_squared_error_by_payout": dict(zip(PAYOUTS, np.mean(list(by_condition.values()), axis=0).tolist())),
        "raw_squared_error_by_condition_and_payout": {
            condition: dict(zip(PAYOUTS, error.tolist())) for condition, error in by_condition.items()
        },
        "win_calibration_bins": calibration["calibration_bins"],
        "nonzero_calibration_error": calibration["diagnostics"]["nonzero_calibration_error"],
    }


def run(data, out, compare_to=None):
    started = perf_counter()
    data, out = Path(data).resolve(), Path(out).resolve()
    if out.exists() or out == data or data in out.parents:
        raise ValidationError("Choose a fresh output directory outside the public data.")
    metadata = round_metadata(data)
    if metadata["round"] != "main":
        raise ValidationError("This experiment compares against a frozen main-round forecast.")
    train = data / "train"
    train_q, train_h = load_inputs(train)
    labels = read_labels(train / "labels.csv", train_q)
    groups = _conditions(train, train_q)
    name, config, cv = _train_validation(train_q, train_h, labels, groups)
    model = BayesianLagPairForecaster(config).fit(train_q, train_h, labels)
    training_seconds = perf_counter() - started

    queries, history, targets, groups, reference = tuning_data(data / "validation")
    ids = [q.query_id for q in queries]
    episodes = np.array([q.episode_id for q in queries])
    conditions = np.array([groups[key] for key in ids])
    counts = {condition: len(set(episodes[conditions == condition])) for condition in CONDITIONS}
    if len(ids) != 384 or len(set(ids)) != 384 or len(set(episodes)) != 48 or set(counts.values()) != {12}:
        raise ValidationError("Expected exactly 48 tuning episodes and 384 queries.")
    lead, lead_hash = load_component(LEAD_CSV, ids)
    logistic, logistic_hash = load_component(DEFAULT_COMPARISON / "two_stage_self.csv", ids)
    signed_saved, signed_hash = load_component(SIGNED_CSV, ids)
    if lead_hash != LEAD_SHA256:
        raise ValidationError("The frozen lead CSV does not match its expected hash.")
    priors = model.prior_rows(queries)
    spins, cutoffs = tick_arrays(queries, history)
    rows = np.arange(len(ids))
    inference_started = perf_counter()
    forecast, posterior = lag_pair_rollout(spins, priors, config, diagnostic_ticks=cutoffs)
    signed, _ = lag_pair_rollout(spins, priors, GRID["signed_48"])
    no_switch, _ = lag_pair_rollout(spins, priors, GRID["signed_static"])
    undiscounted, _ = motif_rollout(spins, priors, MotifConfig("singleton_pair"))
    signed_delta = float(np.max(np.abs(signed[rows, cutoffs] - signed_saved)))
    static_delta = float(np.max(np.abs(no_switch - undiscounted)))
    if max(signed_delta, static_delta) > 1e-12:
        raise ValidationError("The original signed model controls must reproduce the frozen forecasts.")

    prior = np.asarray(model.prior.global_prior)[1:]
    positive = conditional_positive(logistic, prior / prior.sum())
    head_delta = float(np.max(np.abs(with_positive_head(lead[:, 0], positive) - lead)))
    if head_delta > 1e-12:
        raise ValidationError("The lead must use the frozen logistic conditional payout mix.")
    standalone = with_positive_head(forecast[rows, cutoffs, 0], positive)
    fixed = convex_mix(lead, standalone, BLEND_WEIGHT)
    by_id = {row.query_id: row for row in targets}
    truth = np.array([by_id[key].probabilities for key in ids])
    ref = np.array([reference[key] for key in ids])
    folds = list(_train_folds(queries, groups))
    blended, blend_weights, _ = crossfit_blend(lead, standalone, truth, ref, episodes, conditions, folds)
    arrays = {
        "lead": lead,
        "signed_control": with_positive_head(signed[rows, cutoffs, 0], positive),
        "standalone": standalone,
        "fixed_blend": fixed,
        "crossfit_blend": blended,
    }
    inference_seconds = perf_counter() - inference_started

    out.mkdir(parents=True)
    model.prior.save(out / "population_prior.json")
    scores, forecasts, hashes, diagnostics, repeat_deltas = {}, {}, {}, {}, {}
    for variant, array in arrays.items():
        if variant == "lead":
            current, hashes[variant] = lead, lead_hash
        else:
            path = out / (variant + ".csv")
            write_submission(path, dict(zip(ids, array)))
            current, hashes[variant] = load_component(path, ids)
        forecasts[variant] = dict(zip(ids, current))
        scores[variant] = contest_score(forecasts[variant], targets, groups, reference,
                                        max_points=metadata["max_points"], version=VERSION)
        diagnostics[variant] = _probability_diagnostics(forecasts[variant], targets, groups)
        if compare_to is not None and variant != "lead":
            previous, _ = load_component(Path(compare_to) / path.name, ids)
            repeat_deltas[variant] = float(np.max(np.abs(current - previous)))
            if repeat_deltas[variant] > 1e-8:
                raise ValidationError(f"{variant} failed the fresh replay tolerance.")
    if (abs(scores["lead"]["technical_points"] - LEAD_POINTS) > 1e-8
            or abs(scores["lead"]["mean_loss_ratio"] - LEAD_RATIO) > 1e-10
            or abs(scores["signed_control"]["technical_points"] - SIGNED_POINTS) > 1e-8):
        raise ValidationError("Tuning controls do not match the frozen lead and signed-model scores.")
    intervals = {
        variant: paired_ratio_interval(prediction, forecasts["lead"], targets, groups, reference)
        for variant, prediction in forecasts.items() if variant != "lead"
    }
    condition_deltas = {
        variant: {condition: score["by_condition"][condition]["loss_ratio"] -
                    scores["lead"]["by_condition"][condition]["loss_ratio"] for condition in CONDITIONS}
        for variant, score in scores.items() if variant != "lead"
    }
    shortlist = {
        variant: intervals[variant]["delta_r"] <= -.01 and max(condition_deltas[variant].values()) <= .05
        for variant in ("standalone", "fixed_blend")
    }
    report = {
        "round": metadata["round"], "season_id": metadata["season_id"], "split": "tuning",
        "tuning_episodes": sorted(set(episodes)), "tuning_episodes_by_condition": counts,
        "tuning_queries": len(ids), "selected": name, "config": asdict(config), "train_cv": cv,
        "scores": scores, "paired_delta_r_vs_lead": intervals, "condition_deltas_vs_lead": condition_deltas,
        "shortlist": shortlist, "probability_diagnostics": diagnostics,
        "posterior_by_condition": _posterior_summary(posterior, conditions),
        "blend": {"fixed_weight": BLEND_WEIGHT, "crossfit_weights": blend_weights,
                  "held_out_episodes": [sorted(fold) for fold in folds],
                  "crossfit_intervals": "Descriptive, with the out-of-fold forecasts held fixed."},
        "checks": {"signed_control_max_abs_delta": signed_delta, "zero_hazard_max_abs_delta": static_delta,
                   "lead_positive_head_max_abs_delta": head_delta, "replay_max_abs_deltas": repeat_deltas},
        "sha256": {**hashes, "frozen_logistic": logistic_hash, "frozen_signed": signed_hash},
        "seconds": {"grouped_training": training_seconds, "tuning_inference": inference_seconds,
                    "total": perf_counter() - started},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "threads": 2, "command": shlex.join([sys.executable, *sys.argv]),
    }
    write_json(out / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compare-to", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        report = run(args.data, args.out, args.compare_to)
    print(f"Training-selected variant: {report['selected']}")
    for name, row in report["train_cv"]["candidates"].items():
        print(f"  {name:20} cutoff win Brier {row['cutoffs']['brier']:.6f}")
    for name, score in report["scores"].items():
        print(f"{name:20} {score['technical_points']:.6f} points, R={score['mean_loss_ratio']:.6f}")
    print(args.out / "report.json")


if __name__ == "__main__":
    main()
