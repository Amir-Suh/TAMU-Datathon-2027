"""Evaluate two-parent full-distribution stacks on the frozen validation folds."""

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
KIT = HERE.parents[2]
EXPERIMENTS = KIT / "signal_experiments"
REPAIR = KIT / "m5_m6_repair"
FROZEN = REPAIR / "results/repaired-m5-m6"
DATA = KIT / "public/validation"
FOLDS = EXPERIMENTS / "folds.csv"
HYBRID = EXPERIMENTS / "combined/m3_match_split_win_pair_l31_factor_gru_size.csv"
FULL_BLEND = EXPERIMENTS / "01_fredy_replay/forecasts/pair_l31_factor_gru_m6_repaired.csv"
DIAGNOSTICS = EXPERIMENTS / "ceiling_analysis/diagnostics.json"
sys.path[:0] = [str(REPAIR), str(KIT), str(EXPERIMENTS)]

from models.common.crossfit import subset_split
from models.common.data import MISSING
from run_repaired_comparison import input_only
from score import compare, sha256
from slot_forecasting.io import (read_rows, read_submission, read_targets, require_ids, write_json,
                                 write_rows, write_submission)
from slot_forecasting.schema import CONDITIONS, PAYOUTS

ARMS = ("equal", "global", "support", "entropy")
SUPPORT_BOUNDARY = 40
ENTROPY_BOUNDARY = 0.01
SHRINK_EPISODES = 8
SEED = 20261107
DRAWS = 10000


def features(model, split):
    """Project onto the observed prefix before running the frozen independent M3."""
    pre_cutoff = np.arange(split.X.shape[1]) < split.cutoff[:, None]
    causal = replace(split, X=np.where(pre_cutoff, split.X, MISSING),
                     labels=None, conditions=None, targets=None, reference=None)
    posterior = model._infer(causal)["ind"]
    pair_mass = posterior["post"].sum(axis=(2, 3))
    pairs = np.asarray(model.pairs)[pair_mass.argmax(axis=1)]
    mass = np.column_stack([pair_mass, posterior["post0"]])
    mass /= mass.sum(axis=1, keepdims=True)
    entropy = -(mass * np.log(np.maximum(mass, 1e-300))).sum(axis=1) / np.log(mass.shape[1])
    observed = causal.X != MISSING
    ticks = np.arange(observed.shape[1])
    lag_a = ticks - pairs[:, :1]
    lag_b = ticks - pairs[:, 1:]
    usable = observed & (lag_b >= 0)
    usable &= np.take_along_axis(observed, lag_a.clip(0), axis=1)
    usable &= np.take_along_axis(observed, lag_b.clip(0), axis=1)
    return pairs, usable.sum(axis=1), entropy, observed.sum(axis=1)


def group_weights(episodes, conditions, reference_loss):
    """Express the contest objective as one weighted sum over fitting rows."""
    weight = np.zeros(len(episodes))
    scales = {}
    for condition in CONDITIONS:
        rows = conditions == condition
        unique, indexes, counts = np.unique(episodes[rows], return_inverse=True, return_counts=True)
        if len(unique) == 0:
            raise ValueError(f"Missing fitting condition {condition}.")
        episode_weight = 1 / (len(unique) * counts[indexes])
        scale = max(1e-9, float(episode_weight @ reference_loss[rows]))
        weight[rows] = episode_weight / (len(CONDITIONS) * scale)
        scales[condition] = scale
    return weight, scales


def fit_gate(feature, boundary, episodes, curvature, slope, global_weight, shrink):
    bins = []
    for rows in (feature < boundary, feature >= boundary):
        rows = rows & np.isfinite(feature)
        if rows.any():
            leverage = float(curvature[rows].sum())
            weight = (slope[rows].sum() + shrink * global_weight) / (leverage + shrink) if leverage else global_weight
            bins.append({"weight": float(np.clip(weight, 0, 1)), "fit_range": [float(feature[rows].min()),
                                                                            float(feature[rows].max())],
                         "queries": int(rows.sum()), "episodes": len(np.unique(episodes[rows])),
                         "curvature": leverage})
        else:
            bins.append({"weight": global_weight, "fit_range": None, "queries": 0, "episodes": 0,
                         "curvature": 0.0})
    return {"boundary": boundary, "bins": bins}


def fit_weights(hybrid, full_blend, support, entropy, episodes, conditions, truth, reference):
    weight, scales = group_weights(episodes, conditions, np.square(reference - truth).sum(axis=1))
    change = full_blend - hybrid
    curvature = weight * np.square(change).sum(axis=1)
    slope = weight * (change * (truth - hybrid)).sum(axis=1)
    total = float(curvature.sum())
    global_weight = float(np.clip(slope.sum() / total, 0, 1)) if total else 0.5
    shrink = SHRINK_EPISODES * total / len(np.unique(episodes))
    return {
        "global_weight": global_weight,
        "fit_reference_scales": scales,
        "penalty_curvature": shrink,
        "support": fit_gate(support, SUPPORT_BOUNDARY, episodes, curvature, slope, global_weight, shrink),
        "entropy": fit_gate(entropy, ENTROPY_BOUNDARY, episodes, curvature, slope, global_weight, shrink),
    }


def apply_gate(feature, gate, global_weight):
    weights = np.full(len(feature), global_weight)
    used_bin = np.full(len(feature), False)
    for row in gate["bins"]:
        if row["fit_range"] is not None:
            lo, hi = row["fit_range"]
            selected = np.isfinite(feature) & (feature >= lo) & (feature <= hi)
            weights[selected] = row["weight"]
            used_bin |= selected
    return weights, ~used_bin


def predict(hybrid, full_blend, support, entropy, fit):
    global_weight = fit["global_weight"]
    weights = {"equal": np.full(len(hybrid), 0.5), "global": np.full(len(hybrid), global_weight)}
    fallback = {}
    for name, feature in (("support", support), ("entropy", entropy)):
        weights[name], fallback[name] = apply_gate(feature, fit[name], global_weight)
    change = full_blend - hybrid
    predictions = {name: hybrid + weight[:, None] * change for name, weight in weights.items()}
    return predictions, weights, fallback


def fold_run(held, hybrid, full_blend, support, entropy, episodes, conditions, truth, reference):
    fit = ~held
    if set(episodes[fit]) & set(episodes[held]):
        raise ValueError("Episode intersects fitting and held sets.")
    fit_data = (hybrid, full_blend, support, entropy, episodes, conditions, truth, reference)
    coefficients = fit_weights(*(values[fit] for values in fit_data))
    inference_data = (hybrid, full_blend, support, entropy)
    predictions, weights, fallback = predict(*(values[held] for values in inference_data), coefficients)
    return coefficients, predictions, weights, fallback


def partition_scores(predictions, truth, reference, episodes, conditions, mask, points):
    """Recompute episode means and reference denominators within each slice."""
    errors = {name: np.square(p - truth).sum(axis=1) for name, p in predictions.items()}
    errors["reference"] = np.square(reference - truth).sum(axis=1)
    condition_scores = {}
    sampled = {name: [] for name in predictions}
    rng = np.random.default_rng(SEED)
    for condition in CONDITIONS:
        select = mask & (conditions == condition)
        unique, indexes, counts = np.unique(episodes[select], return_inverse=True, return_counts=True)
        if not len(unique):
            continue
        values = {name: np.bincount(indexes, weights=v[select]) / counts for name, v in errors.items()}
        denom = max(1e-9, float(values["reference"].mean()))
        resamples = rng.integers(len(unique), size=(DRAWS, len(unique)))
        ref_draws = values["reference"][resamples].mean(axis=1).clip(1e-9)
        condition_scores[condition] = {name: float(values[name].mean() / denom) for name in predictions}
        for name in predictions:
            sampled[name].append(values[name][resamples].mean(axis=1) / ref_draws)
    ratios = {name: float(np.mean([r[name] for r in condition_scores.values()])) for name in predictions}
    draw_ratios = {name: np.mean(values, axis=0) for name, values in sampled.items()}

    def paired(name, baseline):
        interval = np.quantile(draw_ratios[name] - draw_ratios[baseline], [0.025, 0.975])
        return {"delta_R": ratios[name] - ratios[baseline], "delta_R_95": interval.tolist()}

    return {
        "queries": int(mask.sum()),
        "episodes": len(np.unique(episodes[mask])),
        "conditions": list(condition_scores),
        "models": {
            name: {"R": ratio, "points": points / (1 + ratio),
                   "vs_hybrid": paired(name, "hybrid"),
                   "vs_global": paired(name, "global")}
            for name, ratio in ratios.items()
        },
    }


def verify_features(model, split, original):
    assert all(getattr(split, key) is None for key in ("labels", "targets", "conditions", "reference"))
    rng = np.random.default_rng(SEED)
    order = rng.permutation(split.n)
    reordered_split = subset_split(split, order)
    padded = np.pad(split.X, ((0, 0), (0, 13)), constant_values=MISSING)
    future = padded.copy()
    future_mask = np.arange(padded.shape[1]) >= split.cutoff[:, None]
    future[future_mask] = rng.integers(0, len(PAYOUTS), size=future_mask.sum())
    for altered, reorder in ((reordered_split, order), (replace(split, X=padded), np.arange(split.n)),
                             (replace(split, X=future), np.arange(split.n))):
        for observed, expected in zip(features(model, altered), original):
            np.testing.assert_array_equal(observed, expected[reorder])
    pairs, support, _, _ = original
    for i, (a, b) in enumerate(pairs):
        assert support[i] == sum(np.all(split.X[i, [t, t - a, t - b]] != MISSING)
                                 for t in range(b, split.cutoff[i]))


def verify_fold(held, args, expected):
    fit, predictions, weights, fallback = expected
    changed = [value.copy() for value in args]
    changed[5][held] = "withheld"
    changed[6][held] = np.nan
    changed[7][held] = np.nan
    perturbed_fit, perturbed_pred, _, _ = fold_run(held, *changed)
    assert fit == perturbed_fit
    for arm in ARMS:
        np.testing.assert_array_equal(predictions[arm], perturbed_pred[arm])
    order = np.random.default_rng(SEED).permutation(len(held))
    reordered_fit, reordered_pred, _, _ = fold_run(held[order], *(value[order] for value in args))
    reverse = np.argsort(order[held[order]])
    for arm in ARMS:
        np.testing.assert_allclose(reordered_pred[arm][reverse], predictions[arm], atol=1e-14, rtol=0)
        assert np.all((weights[arm] >= 0) & (weights[arm] <= 1))
        np.testing.assert_allclose(predictions[arm].sum(axis=1), 1, atol=1e-14, rtol=0)
    np.testing.assert_allclose(reordered_fit["global_weight"], fit["global_weight"], atol=1e-14, rtol=0)
    for name in ("support", "entropy"):
        probe = np.array([np.nan, -np.inf, -1, 1e9, np.inf])
        values, used_fallback = apply_gate(probe, fit[name], fit["global_weight"])
        assert used_fallback.all()
        assert (values == fit["global_weight"]).all()
        assert (weights[name][fallback[name]] == fit["global_weight"]).all()


def git_value(*args):
    result = subprocess.run(["git", *args], cwd=KIT, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def main(out):
    start = perf_counter()
    historical = json.loads(DIAGNOSTICS.read_text())
    checkpoint = FROZEN / "models.joblib"
    frozen_manifest = FROZEN / "freeze.json"
    expected_hashes = {DATA / name: value for name, value in historical["data_sha256"].items()}
    expected_hashes.update({
        HYBRID: historical["forecast_sha256"]["best"],
        FULL_BLEND: historical["forecast_sha256"]["full_blend"],
        checkpoint: json.loads(frozen_manifest.read_text())["checkpoint_sha256"],
    })
    for path, expected_hash in expected_hashes.items():
        if sha256(path) != expected_hash:
            raise ValueError(f"Unexpected frozen input {path}.")
    metadata_path = DATA.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    model = joblib.load(checkpoint)["m3"]
    assert not model.share_episode
    split = input_only(DATA.parent, "validation")
    ids, episodes = split.query_ids, split.episode_ids
    derived = features(model, split)
    features_seconds = perf_counter() - start
    verify_features(model, split, derived)
    pairs, support, entropy, observed = derived

    def aligned(mapping):
        require_ids(mapping, ids)
        return np.asarray([mapping[key] for key in ids])

    hybrid = aligned(read_submission(HYBRID, ids))
    full_blend = aligned(read_submission(FULL_BLEND, ids))
    truth_rows = read_targets(DATA / "targets.csv")
    truth_episode = aligned({t.query_id: t.episode_id for t in truth_rows})
    np.testing.assert_array_equal(episodes, truth_episode)
    truth = aligned({t.query_id: t.probabilities for t in truth_rows})
    reference = aligned(read_submission(DATA / "reference.csv", ids))
    conditions = aligned({row["query_id"]: row["condition"] for row in
                          read_rows(DATA / "conditions.csv", ("query_id", "condition"))})
    fold_rows = [row for row in read_rows(FOLDS, ("split", "episode_id", "fold"))
                 if row["split"] == "validation"]
    fold_map = {row["episode_id"]: int(row["fold"]) for row in fold_rows}
    assert len(fold_map) == len(fold_rows) and set(fold_map) == set(episodes)
    folds = np.asarray([fold_map[episode] for episode in episodes])
    assert set(folds) == set(range(4))
    predictions = {name: np.full_like(hybrid, np.nan) for name in ARMS}
    weights = {name: np.full(len(ids), np.nan) for name in ARMS}
    fallback = {name: np.zeros(len(ids), dtype=bool) for name in ("support", "entropy")}
    fits = {}
    args = (hybrid, full_blend, support, entropy, episodes, conditions, truth, reference)
    for fold in range(4):
        held = folds == fold
        assert len(set(episodes[held])) == 24 and held.sum() == 192
        assert all(len(set(episodes[held & (conditions == c)])) == 6 for c in CONDITIONS)
        result = fold_run(held, *args)
        verify_fold(held, args, result)
        fit, fold_predictions, fold_weights, fold_fallback = result
        fits[fold] = {"fit_episodes": sorted(set(episodes[~held])), "held_episodes": sorted(set(episodes[held])),
                      "held_fallback": {name: int(value.sum()) for name, value in fold_fallback.items()},
                      **fit}
        for name in ARMS:
            predictions[name][held] = fold_predictions[name]
            weights[name][held] = fold_weights[name]
        for name in fallback:
            fallback[name][held] = fold_fallback[name]

    out.mkdir(parents=True, exist_ok=True)
    paths = {name: out / ("validation_predictions.csv" if name == "support" else f"validation_{name}.csv")
             for name in ARMS}
    for name, path in paths.items():
        write_submission(path, dict(zip(ids, predictions[name])))

    columns = ("query_id", "episode_id", "fold", "observed_ticks", "lag_a", "lag_b", "usable_lag_triples",
               "posterior_entropy", *[f"{name}_full_blend_weight" for name in ARMS],
               "support_fallback", "entropy_fallback")
    rows = ((key, episodes[i], folds[i], observed[i], *pairs[i], support[i], entropy[i],
             *(weights[name][i] for name in ARMS), int(fallback["support"][i]), int(fallback["entropy"][i]))
            for i, key in enumerate(ids))
    write_rows(out / "query_weights.csv", columns, (dict(zip(columns, row)) for row in sorted(rows)))
    write_json(out / "coefficients.json", fits)
    max_points = metadata["max_points"]
    candidates = {"hybrid": hybrid, "full_blend": full_blend, **predictions}
    slices = {"overall": np.ones(len(ids), dtype=bool), **{c: conditions == c for c in CONDITIONS},
              "support_lt_40": support < 40, "support_ge_40": support >= 40,
              "64_observed_ticks": observed == 64, "complete_history": observed == split.cutoff,
              "top_pair_4_7": (pairs == [4, 7]).all(axis=1)}
    subgroup = {name: partition_scores(candidates, truth, reference, episodes, conditions, mask, max_points)
                for name, mask in slices.items()}
    scores = compare(DATA, HYBRID, {"full_blend": FULL_BLEND, **paths}, max_points, DRAWS)
    assert abs(scores["models"]["control"]["points"] - 56.341022) < 5e-7
    assert abs(scores["models"]["full_blend"]["points"] - 56.064085) < 5e-7
    for name, model_scores in scores["models"].items():
        summary = subgroup["overall"]["models"]["hybrid" if name == "control" else name]
        assert abs(model_scores["R"] - summary["R"]) < 1e-14
    write_json(out / "scores.json", scores)
    write_json(out / "subgroup_scores.json", subgroup)
    plan = {
        "scope": "Reused main-validation development with episode-held-out stack weights and frozen parents. "
                 "Shared transductive Fredy components trained on pre-cutoff observed ticks from all main splits.",
        "metadata": metadata,
        "classes": PAYOUTS,
        "parent_order": ["hybrid", "full_blend"],
        "input_sha256": {str(p.relative_to(KIT)): sha256(p) for p in
                         [*expected_hashes, FOLDS, frozen_manifest, metadata_path, DIAGNOSTICS]},
        "source_sha256": {str(p.relative_to(KIT)): sha256(p) for p in
                          [Path(__file__), EXPERIMENTS / "score.py", REPAIR / "run_repaired_comparison.py",
                           REPAIR / "models/m3_lag_pair_flip/model.py", KIT / "slot_forecasting/contest.py",
                           KIT / "slot_forecasting/scoring.py"]},
        "git_branch": git_value("branch", "--show-current"),
        "git_commit": git_value("rev-parse", "--verify", "HEAD"),
        "runtime": {"python": platform.python_version(), "numerical_thread_limit": 2,
                    **{p: importlib.metadata.version(p) for p in
                       ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")}},
        "support_boundary": SUPPORT_BOUNDARY,
        "entropy_boundary": ENTROPY_BOUNDARY,
        "entropy_feature": "Normalized Shannon entropy of the M3 pair posterior including the null.",
        "shrink_pseudo_episodes": SHRINK_EPISODES,
        "fallback": "Global weight for nonfinite features or outside the fitted range of their bin.",
        "bootstrap": {"draws": DRAWS, "seed": SEED, "unit": "paired whole episodes stratified by condition",
                      "refits": False},
        "checks_passed": ["frozen input hashes", "published fold membership and condition balance",
                          "independent cutoff-prefix features with scalar usable triple counts",
                          "feature row permutation, missing padding, and injected post-cutoff outcomes",
                          "stack row permutation and held-target/condition/reference poisoning",
                          "fallback on missing and out-of-range features",
                          "convex weights and normalized rows", "official score and baseline replay"],
        "seconds_before_verification_to_features": features_seconds,
        "seconds_elapsed": perf_counter() - start,
    }
    write_json(out / "plan.json", plan)
    for name, result in scores["models"].items():
        print(f"{name:12} {result['points']:.6f} R={result['R']:.8f} ΔR={result['delta_R']:+.8f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE)
    opts = parser.parse_args()
    with threadpool_limits(limits=2):
        main(opts.out.resolve())
