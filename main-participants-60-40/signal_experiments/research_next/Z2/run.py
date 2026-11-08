"""Fit and compare static and sticky payout states on training histories."""
import argparse
from dataclasses import replace
import importlib
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                 "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"

import numpy as np
import scipy
from scipy.optimize import check_grad
from scipy.special import expit, logit
from sklearn import __version__ as sklearn_version
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
KIT = HERE.parents[2]
EXPERIMENTS = KIT / "signal_experiments"
PUBLIC = KIT / "public"
FOLDS = EXPERIMENTS / "folds.csv"
BASE = EXPERIMENTS / "combined/m3_match_split_win_pair_l31_factor_gru_size.csv"
sys.path[:0] = [str(KIT), str(KIT / "m5_m6_repair")]

from models.common.crossfit import subset_split
from models.common.data import MISSING, TYPES
from run_repaired_comparison import input_only
from signal_experiments.research_next.synthetic import generate, mask_history, TAIL
from signal_experiments.score import compare, sha256
from slot_forecasting.io import read_rows, read_submission, write_json, write_rows, write_submission
from slot_forecasting.schema import PAYOUTS
from state import (
    History, PRIOR_SD, fit, forward, history_prior, infer, joint_forecast, objective, parameters,
)

w1 = importlib.import_module("signal_experiments.research_next.W1.run")
ceiling = importlib.import_module("signal_experiments.ceiling_analysis.analyze")
DURATION = (None, 16, 64, 128)
ARMS = {(dwell, joint): f"{'static' if dwell is None else f'sticky_{dwell}'}_{'joint' if joint else 'payout'}"
        for dwell in DURATION for joint in (False, True)}


def prefix_only(split):
    before = np.arange(split.X.shape[1])[None, :] < split.cutoff[:, None]
    return replace(split, X=np.where(before, split.X, MISSING), labels=None, targets=None,
                   reference=None, conditions=None)


def frozen_structure_inputs(split):
    # The recorded signal applies at the actual cutoff. Older ticks have no signal.
    return replace(prefix_only(split), signal=np.zeros(split.n, dtype=int),
                   labels=np.full(split.n, MISSING, dtype=np.int8))


def aligned_lags(split, pairs):
    rows = np.arange(split.n)[:, None]
    ticks = np.arange(split.X.shape[1])[None, :]
    a, b = ticks - pairs[:, :1], ticks - pairs[:, 1:]
    wa = split.X[rows, a.clip(0)]
    wb = split.X[rows, b.clip(0)]
    return (wa > 0) != (wb > 0), (b >= 0) & (wa >= 0) & (wb >= 0)


def lag_baseline(model, split):
    """Use a prefix plug-in MAP lag/flip hypothesis for historical emissions."""
    frame = frozen_structure_inputs(split)
    result = model._infer(frame)["ind"]
    post = result["post"]
    best = post.reshape(split.n, -1).argmax(axis=1)
    pair_index, direction, tau_index = np.unravel_index(best, post.shape[1:])
    best_mass = post[np.arange(split.n), pair_index, direction, tau_index]
    use_null = result["post0"] > best_mass
    pairs = np.asarray(model.pairs)[pair_index]
    change = np.r_[0, model.tau_grid][tau_index]
    ticks = np.arange(split.X.shape[1])[None, :]
    flip = (change[:, None] > 0) & (ticks < change[:, None])
    high = direction[:, None].astype(bool) ^ flip
    differs, known = aligned_lags(frame, pairs)
    ctx = frame.context
    active = np.where(differs == high, model.hi[ctx, None], model.lo[ctx, None])
    rate = np.where(known & ~use_null[:, None], active, model.win_prior[ctx, None])
    selected_pairs = np.where(use_null[:, None], 0, pairs)
    return History.build(frame, rate.clip(.005, .995)), selected_pairs, np.maximum(best_mass, result["post0"])


def model_for_history(split):
    return w1.refit_m3(frozen_structure_inputs(split), tune=False)


def describe(theta, iterations):
    intercept, q1, _, shift = parameters(theta)
    return {"history_win_intercept_by_type": dict(zip(TYPES, intercept.tolist())),
            "conditional_1x_by_type_low_high": dict(zip(TYPES, q1.tolist())),
            "state_high_vs_low_win_logit_shift": float(shift),
            "parameter_vector": theta.tolist(),
            "optimizer_iterations": int(iterations)}


def train_folds(train):
    rows = [row for row in read_rows(FOLDS, ("split", "episode_id", "fold")) if row["split"] == "train"]
    by_episode = {row["episode_id"]: int(row["fold"]) for row in rows}
    if len(by_episode) != len(rows) or set(by_episode) != set(train.episode_ids):
        raise ValueError("Training fold IDs disagree with observed histories.")
    return np.array([by_episode[ep] for ep in train.episode_ids])


def select_dwell(train):
    folds = train_folds(train)
    scores = {name: [] for name in ARMS.values()}
    for fold in sorted(set(folds)):
        fitting = subset_split(train, np.flatnonzero(folds != fold))
        held = subset_split(train, np.flatnonzero(folds == fold))
        rule = model_for_history(fitting)
        fitting_history, _, _ = lag_baseline(rule, fitting)
        held_history, _, _ = lag_baseline(rule, held)
        for (dwell, joint), name in ARMS.items():
            theta, _ = fit(fitting_history, dwell, joint)
            _, nll = infer(held_history, theta, dwell)
            episode_losses = [nll[held.episode_ids == ep].sum() /
                              held_history.count[held.episode_ids == ep].sum()
                              for ep in sorted(set(held.episode_ids))]
            scores[name].append(float(np.mean(episode_losses)))
        print(f"train fold {fold} complete", flush=True)
    selected = {family: min(DURATION[1:], key=lambda d: np.mean(scores[ARMS[d, joint]]))
                for joint, family in ((False, "payout"), (True, "joint"))}
    return selected, scores


def tail_ratios(train):
    hist = train.X
    counts = np.array([[np.count_nonzero(hist[train.machine_type == i] == size)
                        for size in range(2, 5)] for i in range(len(TYPES))])
    return counts / counts.sum(axis=1, keepdims=True)


def tiny_checks():
    emission = np.array([[.6, .2], [1, 1], [.3, .8], [.4, .5]])[:, None]
    switch = 1 / 16
    prior, filtered, scale = forward(emission, switch)
    path_total, end = 0, np.zeros(2)
    for path in itertools.product((0, 1), repeat=len(emission)):
        mass = .5
        for tick, state in enumerate(path):
            mass *= emission[tick, 0, state]
            if tick:
                mass *= switch if state != path[tick - 1] else 1 - switch
        path_total += mass
        end[path[-1]] += mass
    assert np.allclose(scale.prod(), path_total, atol=1e-14, rtol=0)
    assert np.allclose(filtered[-1, 0], end / path_total, atol=1e-14, rtol=0)
    assert np.allclose(prior[2, 0], prior[1, 0] * (1 - switch) + prior[1, 0, ::-1] * switch)
    swap_prior, swap_filtered, swap_scale = forward(emission[..., ::-1], switch)
    assert np.allclose(swap_prior[..., ::-1], prior) and np.allclose(swap_scale, scale)
    assert np.allclose(swap_filtered[..., ::-1], filtered)
    base = np.array([.2, .7])
    pr = np.array([[.3, .7], [.5, .5]])
    q1 = np.array([[.3, .5], [.7, .8]])
    tails = np.tile([.3, .4, .3], (2, 1))
    p = joint_forecast(base, pr, q1, -.4, tails)
    assert np.allclose(p, joint_forecast(base, pr[:, ::-1], q1[:, ::-1], .4, tails))
    assert np.allclose(1 - joint_forecast(base, pr, q1, 0, tails)[:, 0], base, atol=1e-14)
    assert np.allclose(p.sum(axis=1), 1, atol=1e-14)
    return {"brute_force_likelihood_abs_error": float(abs(scale.prod() - path_total)),
            "label_swap": True, "missing_tick_transition": True, "joint_normalization": True,
            "payout_only_win_invariance": True}


def synthetic_baseline(fixture, split):
    frame = prefix_only(split)
    differs, known = aligned_lags(frame, fixture.pairs)
    kind = frame.machine_type
    win = np.where(differs == fixture.polarity[:, None],
                   np.array([.95, .84, .59])[kind, None],
                   np.array([.16, .055, .015])[kind, None])
    return expit(logit(np.where(known, win, .44)) + .35 * frame.signal[:, None])


def synthetic_run():
    checks = tiny_checks()
    results = {}
    for mode in ("static", "sticky", "iid"):
        train = generate(episodes=24, seed=4721, mode=mode)
        held = generate(episodes=24, seed=4927, mode=mode)
        history = History.build(train.split, synthetic_baseline(train, train.split))
        if mode == "static":
            short = replace(train.split, X=train.split.X[:4, :20], cutoff=np.full(4, 20),
                            machine_type=train.split.machine_type[:4])
            small_history = History.build(short, synthetic_baseline(train, train.split)[:4, :20])
            prior_mean = history_prior(small_history)
            gradient_error = check_grad(lambda p: objective(p, small_history, 1 / 16, prior_mean)[0],
                                        lambda p: objective(p, small_history, 1 / 16, prior_mean)[1],
                                        prior_mean)
            if gradient_error > 1e-4:
                raise ValueError(f"State likelihood gradient check failed: {gradient_error}.")
            checks["likelihood_gradient_finite_difference_error"] = float(gradient_error)
        results[mode] = {}
        for (dwell, joint), name in ARMS.items():
            theta, iterations = fit(history, dwell, joint)
            _, q1, _, shift = parameters(theta)
            entry = {"fit": describe(theta, iterations)}
            for coverage, split in (("full", held.split), ("64_observed", mask_history(held.split, 24))):
                rate = synthetic_baseline(held, split)
                hist = History.build(split, rate)
                prior, nll = infer(hist, theta, dwell)
                row = np.arange(split.n)
                true_state = held.state[row, split.cutoff]
                prediction = joint_forecast(rate[row, split.cutoff], prior, q1[split.machine_type],
                                            shift, TAIL[split.machine_type])
                entry[coverage] = {
                    "queries": split.n,
                    "history_nll_per_observation": float(nll.sum() / hist.count.sum()),
                    "cutoff_five_class_sse": float(np.square(prediction - split.targets).sum(axis=1).mean()),
                    "cutoff_p0_mse": float(np.square(prediction[:, 0] - split.targets[:, 0]).mean()),
                    "cutoff_state_auc": float(roc_auc_score(true_state, prior[:, 1])),
                }
            results[mode][name] = entry
        print(f"synthetic {mode} complete", flush=True)
    return {"checks": checks, "results": results}


def main_round(out):
    train = input_only(PUBLIC, "train")
    validation = input_only(PUBLIC, "validation")
    selected, fold_scores = select_dwell(train)
    rule = model_for_history(train)
    fitting_history, _, _ = lag_baseline(rule, train)
    held_history, pairs, pair_mass = lag_baseline(rule, validation)
    # Insert observations at and after each cutoff, then verify the full historical
    # design remains identical. The parent forecast is frozen and unaffected.
    after = np.arange(validation.X.shape[1])[None, :] >= validation.cutoff[:, None]
    altered = replace(validation, X=np.where(after, 4, validation.X),
                      labels=np.full(validation.n, 4), targets=np.full((validation.n, 5), .2))
    perturbed, _, _ = lag_baseline(rule, altered)
    if not (np.array_equal(held_history.payout, perturbed.payout)
            and np.array_equal(held_history.base_logit, perturbed.base_logit)):
        raise ValueError("Historical design depends on a cutoff or future payout.")

    raw = read_submission(BASE, validation.query_ids)
    raw_win = np.array([sum(raw[key][1:]) for key in validation.query_ids])
    tail = tail_ratios(train)
    arms, paths, fits, features = {}, {}, {}, {}
    for (dwell, joint), name in ARMS.items():
        theta, iterations = fit(fitting_history, dwell, joint)
        fits[name] = describe(theta, iterations)
        _, q1, _, shift = parameters(theta)
        prior, _ = infer(held_history, theta, dwell)
        features[name] = prior[:, 1]
        pred = joint_forecast(raw_win, prior, q1[validation.machine_type], shift,
                              tail[validation.machine_type])
        if not np.allclose(pred.sum(axis=1), 1, atol=1e-14, rtol=0):
            raise ValueError(f"Invalid joint distribution for {name}.")
        if not joint and not np.allclose(1 - pred[:, 0], raw_win, atol=1e-14, rtol=0):
            raise ValueError(f"Payout-only arm {name} altered win mass.")
        arms[name] = pred
        paths[name] = out / f"{name}.csv"
        write_submission(paths[name], dict(zip(validation.query_ids, pred)))

    points = json.loads((PUBLIC / "metadata.json").read_text())["max_points"]
    primary = compare(PUBLIC / "validation", BASE, paths, points)
    matched = {}
    for joint, family in ((False, "payout"), (True, "joint")):
        selected_name = ARMS[selected[family], joint]
        matched[f"selected_sticky_vs_static_{family}"] = compare(
            PUBLIC / "validation", paths[ARMS[None, joint]], {selected_name: paths[selected_name]}, points)
    matched["joint_vs_payout_same_dwell"] = {}
    for dwell in (None, selected["joint"]):
        name = ARMS[dwell, True]
        matched["joint_vs_payout_same_dwell"][name] = compare(
            PUBLIC / "validation", paths[ARMS[dwell, False]], {name: paths[name]}, points)
    reports = [primary, *(report for report in matched.values() if "models" in report),
               *matched["joint_vs_payout_same_dwell"].values()]
    for report in reports:
        report["data"] = "public/validation"
        for model in report["models"].values():
            model["path"] = Path(model["path"]).name
    split = ceiling.load_split(PUBLIC, "validation")
    if split.query_ids != validation.query_ids:
        raise ValueError("Scoring slices disagree with forecast query order.")
    truth = split.targets
    ref = np.square(split.reference - truth).sum(axis=1)
    weights, _ = ceiling.weights(split.episode_ids, split.conditions, ref)
    raw_array = np.array([raw[key] for key in split.query_ids])
    base_error = np.square(raw_array - truth).sum(axis=1)
    slice_scores = {}
    for label, mask in {"short_history": split.conditions == "short_history",
                        "inferred_pair_4_7": (pairs == [4, 7]).all(axis=1)}.items():
        diffs = {name: (np.square(p - truth).sum(axis=1) - base_error) * mask for name, p in arms.items()}
        intervals = ceiling.paired_intervals(split, ref, diffs, np.zeros(split.n))
        slice_scores[label] = {
            name: {"delta_R_contribution": float(weights @ delta),
                   "delta_R_contribution_95": intervals[name]["delta_R_95"],
                   "p0_mse": float(np.square(arms[name][mask, 0] - truth[mask, 0]).mean()),
                   "raw_p0_mse": float(np.square(raw_array[mask, 0] - truth[mask, 0]).mean()),
                   "queries": int(mask.sum())}
            for name, delta in diffs.items()
        }
    rows = [dict(query_id=q, inferred_pair=f"{pair[0]}-{pair[1]}" if pair[0] else "null",
                 map_hypothesis_mass=mass,
                 **{name: p[i] for name, p in features.items()})
            for i, (q, pair, mass) in enumerate(zip(validation.query_ids, pairs, pair_mass))]
    write_rows(out / "state_posteriors.csv", tuple(rows[0]), rows)
    chosen = out / "validation_predictions.csv"
    chosen.write_bytes(paths[ARMS[selected["joint"], True]].read_bytes())
    return {"vs_frozen_hybrid": primary, "matched": matched, "focal_slices": slice_scores,
            "history_train_cv_episode_mean_nll_per_observation": fold_scores,
            "selected_sticky_dwell": selected, "fits": fits, "cutoff_future_perturbation": True,
            "train_tail_ratios_by_type": dict(zip(TYPES, tail.tolist()))}


def manifest():
    metadata = json.loads((PUBLIC / "metadata.json").read_text())
    git = {}
    for name, args in {"branch": ["branch", "--show-current"],
                       "commit": ["rev-parse", "--verify", "HEAD"]}.items():
        process = subprocess.run(["git", *args], cwd=KIT, text=True, capture_output=True)
        git[name] = process.stdout.strip() or None
    files = [BASE, FOLDS, PUBLIC / "metadata.json", Path(__file__), HERE / "state.py",
             Path(w1.__file__), Path(ceiling.__file__), EXPERIMENTS / "score.py",
             HERE.parent / "synthetic.py"]
    files.extend(PUBLIC / split / name for split, names in {
        "train": ("queries.csv", "history.csv"),
        "validation": ("queries.csv", "history.csv", "targets.csv", "reference.csv", "conditions.csv"),
    }.items() for name in names)
    return {
        "experiment": "Z2", "round": metadata["round"], "season_id": metadata["season_id"], "git": git,
        "scope": "Main-development cutoff comparison after fit and dwell selection on train histories. "
                 "History likelihood conditions on a prefix plug-in pair, direction, and flip estimate.",
        "payout_order": list(PAYOUTS),
        "arms": list(ARMS.values()),
        "training": {
            "symmetric_dwell_ticks": list(DURATION),
            "selection": "Held-episode mean history negative log likelihood, per observed tick. "
                         "Upstream M3 parameters are refit in each train fold.",
            "gaussian_prior_sd_in_parameter_order": PRIOR_SD.tolist(),
            "initialization": "Type-pooled 1x rates from fitting wins and a 1.0 ordered logit gap.",
            "historical_signal": "Zeroed, because only the actual cutoff signal is available.",
        },
        "simulation": {"train_seed": 4721, "test_seed": 4927, "episodes_each": 24,
                       "modes": ["static", "sticky", "iid"]},
        "max_cpu_threads": 2,
        "bootstrap": {"draws": 10000, "seed": 20261107, "unit": "condition-stratified episode"},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "scikit_learn": sklearn_version},
        "input_sha256": {str(path.relative_to(KIT)): sha256(path) for path in files},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    start = perf_counter()
    if args.smoke:
        fixture = generate(episodes=12, mode="sticky")
        hist = History.build(fixture.split, synthetic_baseline(fixture, fixture.split))
        theta, iterations = fit(hist, 64, True)
        print(json.dumps({"episodes": 12, "seconds": perf_counter() - start, "checks": tiny_checks(),
                          "fit": describe(theta, iterations)}, indent=2))
        return

    args.out.mkdir(parents=True, exist_ok=True)
    synthetic = synthetic_run()
    write_json(args.out / "synthetic_scores.json", synthetic)
    write_json(args.out / "scores.json", main_round(args.out))
    write_json(args.out / "plan.json", manifest())
    print(f"Z2 synthetic and main validation complete in {perf_counter() - start:.2f}s.")


if __name__ == "__main__":
    main()
