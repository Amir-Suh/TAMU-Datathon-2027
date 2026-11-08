"""Compare prespecified forecast blends and episode-cross-fitted scalar blends."""

import argparse
import hashlib
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from run_model_comparison import DEFAULT_PUBLIC, _train_folds, tuning_data
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.combine_forecasts import (
    conditional_positive, convex_mix, crossfit_blend, fitted_weight, load_component,
    official_row_weights, with_positive_head,
)
from slot_forecasting.contest import contest_score
from slot_forecasting.io import load_inputs, read_labels, read_submission, write_json, write_submission
from slot_forecasting.rounds import VERSION, round_metadata
from slot_forecasting.schema import PROBABILITY_COLUMNS, ValidationError
from slot_forecasting.tuning_metrics import paired_ratio_interval


KIT = Path(__file__).resolve().parent
SEED = 314159
THREADS = 2
COMPONENTS = {
    "adaptive_48": "runs/model-comparison/adaptive_lag.csv",
    "self_logistic": "runs/model-comparison/two_stage_self.csv",
    "adaptive_3072": "runs/adaptive-weight-rate/main/tuning_predictions.csv",
    "signed_motifs": "runs/01-signed-motifs/replay/tuning_predictions.csv",
    "boosted_3072": "runs/04-boosted-correction/main-rate3072/tuning_predictions.csv",
}


def verify_scalar():
    base = np.array([[.7, .2, .07, .02, .01], [.4, .3, .2, .09, .01]])
    alternative = np.array([[.4, .4, .1, .09, .01], [.6, .25, .1, .04, .01]])
    truth = convex_mix(base, alternative, .37)
    weights = np.array([1., 3.])
    fitted = fitted_weight(base, alternative, truth, weights)
    grid = np.linspace(0, 1, 1001)
    grid_loss = (np.square(convex_mix(base, alternative, grid[:, None, None]) - truth).sum(axis=2)) @ weights
    np.testing.assert_allclose(fitted, grid[np.argmin(grid_loss)], atol=1e-12, rtol=0)
    assert fitted_weight(base, base, truth, weights) == 0
    assert fitted_weight(base, alternative, convex_mix(base, alternative, 1.2), weights) == 1
    assert fitted_weight(base, alternative, convex_mix(base, alternative, -.2), weights) == 0
    np.testing.assert_array_equal(convex_mix(base, alternative, 0), base)
    np.testing.assert_allclose(convex_mix(base, alternative, 1), alternative, atol=1e-15, rtol=0)
    prior = np.array([.4, .3, .2, .1])
    all_zero = np.array([[1., 0., 0., 0., 0.]])
    conditional = conditional_positive(all_zero, prior)
    np.testing.assert_array_equal(conditional[0], prior)
    np.testing.assert_array_equal(with_positive_head(all_zero[:, 0], conditional), all_zero)
    return {"dense_grid_scalar": True, "zero_denominator_and_clipping": True,
            "convex_endpoints_and_zero_win_fallback": True}


def run(data, output, paths, compare_to):
    started = perf_counter()
    public, output = data.resolve(), output.resolve()
    if output.exists() or output == public or public in output.parents:
        raise ValidationError("Choose a fresh output directory outside the public data.")
    checks = verify_scalar()
    metadata = round_metadata(public)
    queries, _, targets, groups, reference = tuning_data(public / "validation")
    episodes = sorted({q.episode_id for q in queries})
    expected_episodes = len(metadata["conditions"]) * ((metadata["counts_per_condition"]["validation"] + 1) // 2)
    if (len(episodes), len(queries)) != (expected_episodes, expected_episodes * metadata["machines_per_episode"]):
        raise ValidationError("The tuning manifest does not match the release metadata.")
    ids = sorted(groups)
    target_map = {target.query_id: target for target in targets}
    episode_by_row = np.array([target_map[key].episode_id for key in ids])
    condition_by_row = np.array([groups[key] for key in ids])
    truth = np.array([target_map[key].probabilities for key in ids])
    ref = np.array([reference[key] for key in ids])
    folds = list(_train_folds(queries, groups))
    component_data = {name: load_component(path, ids) for name, path in paths.items()}
    components = {name: data for name, (data, _) in component_data.items()}

    train = public / "train"
    train_q, train_h = load_inputs(train)
    prior = FrequencyForecaster("context").fit(train_q, train_h, read_labels(train / "labels.csv", train_q))
    positive_prior = np.array(prior.global_prior[1:])
    positive_prior /= positive_prior.sum()

    legacy = components["adaptive_48"]
    adaptive_size = conditional_positive(legacy, positive_prior)
    logistic = components["self_logistic"]
    logistic_size = conditional_positive(logistic, adaptive_size)
    logistic_head = {name: with_positive_head(component[:, 0], logistic_size)
                     for name, component in components.items() if name != "self_logistic"}
    for name, swapped in logistic_head.items():
        np.testing.assert_array_equal(swapped[:, 0], components[name][:, 0])
    legacy_swap = logistic_head["adaptive_48"]

    motifs = components["signed_motifs"]
    current = components["adaptive_3072"]
    boosted = components["boosted_3072"]
    motifs_with_current_payout = with_positive_head(motifs[:, 0], conditional_positive(current, adaptive_size))
    motifs_with_boosted_payout = with_positive_head(motifs[:, 0], conditional_positive(boosted, adaptive_size))
    fixed_half = convex_mix(current, motifs_with_current_payout, .5)
    fixed = {**components,
             "legacy_logistic_head_swap": legacy_swap,
             "legacy_half_positive_mix": with_positive_head(legacy[:, 0], convex_mix(adaptive_size, logistic_size, .5)),
             "legacy_whole_90_10": convex_mix(legacy, logistic, .1),
             "adaptive_3072_logistic_head": logistic_head["adaptive_3072"],
             "signed_motifs_logistic_head": logistic_head["signed_motifs"],
             "boosted_3072_logistic_head": logistic_head["boosted_3072"],
             "current_motifs_fixed_25": convex_mix(current, motifs_with_current_payout, .25),
             "current_motifs_fixed_50": fixed_half,
             "current_motifs_fixed_75": convex_mix(current, motifs_with_current_payout, .75),
             "current_motifs_fixed_50_logistic_head": with_positive_head(fixed_half[:, 0], logistic_size)}
    np.testing.assert_array_equal(fixed["legacy_half_positive_mix"][:, 0], legacy[:, 0])
    checks["head_swaps_preserve_p_zero"] = True

    strategies = {
        "legacy_positive_scalar": (legacy, legacy_swap, "adaptive_48"),
        "current_motifs_win_scalar": (current, motifs_with_current_payout, "adaptive_3072"),
        "boosted_motifs_win_scalar": (boosted, motifs_with_boosted_payout, "boosted_3072"),
    }
    oof, in_sample, blend_weights, matched = {}, {}, {}, {}
    permutation = np.random.default_rng(SEED).permutation(len(ids))
    unpermute = np.argsort(permutation)
    for name, (base, alternative, control_name) in strategies.items():
        forecast, weights, final = crossfit_blend(base, alternative, truth, ref, episode_by_row, condition_by_row, folds)
        permuted, permuted_weights, permuted_final = crossfit_blend(
            *(array[permutation] for array in (base, alternative, truth, ref, episode_by_row, condition_by_row)), folds,
        )
        np.testing.assert_allclose(forecast, permuted[unpermute], atol=1e-12, rtol=0)
        np.testing.assert_allclose([*weights, final], [*permuted_weights, permuted_final], atol=1e-12, rtol=0)
        for fold in folds:
            held = np.isin(episode_by_row, sorted(fold))
            changed_truth = truth.copy()
            changed_truth[held] = changed_truth[held, ::-1]
            changed, _, _ = crossfit_blend(base, alternative, changed_truth, ref, episode_by_row, condition_by_row, folds)
            np.testing.assert_array_equal(forecast[held], changed[held])
        oof[name] = forecast
        in_sample[name] = convex_mix(base, alternative, final)
        blend_weights[name] = {"out_of_fold": weights, "full_tuning_in_sample": final}
        matched[name] = control_name
    np.testing.assert_array_equal(oof["legacy_positive_scalar"][:, 0], legacy[:, 0])
    for name, (_, _, control_name) in strategies.items():
        if name != "legacy_positive_scalar":
            swapped_name = f"{name}_logistic_head"
            oof[swapped_name] = with_positive_head(oof[name][:, 0], logistic_size)
            matched[swapped_name] = control_name
    checks.update(row_order_invariant=True, held_out_targets_do_not_affect_fold_forecasts=True)

    def score(prediction, targets=targets):
        return contest_score(prediction, targets, groups, reference, max_points=metadata["max_points"], version=VERSION)

    legacy_prediction = dict(zip(ids, legacy))
    legacy_score = score(legacy_prediction)
    head_swap_score = score(dict(zip(ids, legacy_swap)))
    if metadata["round"] == "main" and (abs(legacy_score["technical_points"] - 33.224516) > 1e-5
                                         or abs(head_swap_score["technical_points"] - 33.374) > .002):
        raise ValidationError("The legacy control or logistic payout-head swap did not reproduce.")
    row_weights = official_row_weights(truth, ref, episode_by_row, condition_by_row)
    np.testing.assert_allclose(row_weights @ np.square(legacy - truth).sum(axis=1),
                               legacy_score["mean_loss_ratio"], atol=1e-12, rtol=0)
    shuffled_predictions = {ids[index]: legacy[index] for index in permutation}
    shuffled_targets = tuple(target_map[ids[index]] for index in permutation)
    if score(shuffled_predictions, shuffled_targets) != legacy_score:
        raise ValidationError("The official score changed after shuffling rows.")
    checks.update(official_weight_objective=True, official_score_order_invariant=True)

    output.mkdir(parents=True)
    csv_hashes, scored, replay_deltas = {}, {}, {}
    fixed_controls = {"adaptive_3072_logistic_head": "adaptive_3072",
                      "signed_motifs_logistic_head": "signed_motifs",
                      "boosted_3072_logistic_head": "boosted_3072",
                      **{name: "adaptive_3072" for name in fixed if name.startswith("current_motifs_fixed_")}}
    for category, candidates in (("fixed", fixed), ("out_of_fold", oof), ("in_sample", in_sample)):
        scored[category] = {}
        for name, array in candidates.items():
            path = output / f"{category}__{name}.csv"
            write_submission(path, dict(zip(ids, array)))
            prediction = read_submission(path, ids)
            csv_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            result = score(prediction)
            errors = np.square(np.array([prediction[key] for key in ids]) - truth)
            result["raw_mean_squared_error_by_payout"] = dict(zip(PROBABILITY_COLUMNS, errors.mean(axis=0).tolist()))
            result["predictions"] = path.name
            if category != "in_sample":
                result["paired_delta_r_vs_adaptive_48"] = paired_ratio_interval(
                    prediction, legacy_prediction, targets, groups, reference,
                )
                control_name = (fixed_controls if category == "fixed" else matched).get(name)
                if control_name and control_name != "adaptive_48":
                    result["matched_control"] = control_name
                    result["paired_delta_r_vs_matched_control"] = paired_ratio_interval(
                        prediction, dict(zip(ids, components[control_name])), targets, groups, reference,
                    )
            scored[category][name] = result
            if compare_to is not None:
                replay = read_submission(compare_to / path.name, ids)
                replay_deltas[path.name] = max(abs(a - b) for key in ids for a, b in zip(prediction[key], replay[key]))

    if replay_deltas and max(replay_deltas.values()) > 1e-8:
        raise ValidationError("A forecast differed from the independent repeat.")
    checks["repeat_normalized_max_difference"] = replay_deltas
    strongest = {category: max(entries, key=lambda name: entries[name]["technical_points"])
                 for category, entries in scored.items()}
    report = {
        "round": metadata["round"], "season_id": metadata["season_id"], "split": "tuning",
        "tuning_episodes": episodes, "tuning_queries": len(ids), "fold_seed": SEED,
        "folds": [{"held_out": sorted(fold), "fit": sorted(set(episodes) - fold)} for fold in folds],
        "settings": {"legacy_logistic_payout_weights": [0, .5, 1], "legacy_whole_logistic_weight": .1,
                     "fixed_signed_motif_win_weights": [.25, .5, .75], "threads": THREADS},
        "scalar_weights": blend_weights, "strongest_by_category": strongest, "scores": scored,
        "scope": {"fixed": "Exploratory tuning comparisons on the released tuning split.",
                  "out_of_fold": "Scalars fitted on other tuning folds; intervals are descriptive and do not refit weights.",
                  "in_sample": "Scalars fitted and scored on all tuning episodes; excluded from out-of-fold scores."},
        "checks": checks, "total_seconds": perf_counter() - started,
        "positive_fallback_training_prior": positive_prior.tolist(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scikit-learn": sklearn.__version__},
        "components": {name: {"path": str(paths[name]), "sha256": digest}
                       for name, (_, digest) in component_data.items()},
        "prediction_sha256": csv_hashes,
    }
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compare-to", type=Path)
    for name, path in COMPONENTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=KIT / path)
    args = parser.parse_args()
    with threadpool_limits(limits=THREADS):
        report = run(args.data, args.out, {name: getattr(args, name) for name in COMPONENTS}, args.compare_to)
    for category, name in report["strongest_by_category"].items():
        points = report["scores"][category][name]["technical_points"]
        print(f"{category}: {name}, {points:.6f} tuning points")
    print(args.out / "report.json")


if __name__ == "__main__":
    main()
