"""E5 staged known-law fit, prediction, and independent score."""
import argparse
import csv
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from filtering import BUDGET, HAZARD, PAIRS, STRENGTH, Head, bocpd, discover, simple, warm
from fixtures import (CUTOFF, EVAL_SEED, EXPERIMENTS, FIT_SEED, MASKS, RARE,
                      SCENARIOS, SMOKE_SEED, generate, masked)
from tree import feature_arrays, ordering, train_trees

HERE = Path(__file__).resolve().parent
KIT = EXPERIMENTS.parent
CHECKPOINT = KIT / "m5_m6_repair/results/repaired-m5-m6/models.joblib"
C1 = EXPERIMENTS / "research_next/C1/run.py"
spec = importlib.util.spec_from_file_location("e5_c1", C1)
c1 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = c1
spec.loader.exec_module(c1)
PAIR_ID = {tuple(p): i for i, p in enumerate(PAIRS)}
REFERENCES = {
    "static_discovered": "static_main", "static_refit": "static_discovered",
    "single_refit": "static_refit", "single_adaptive": "single_refit",
    "multi_flip": "single_adaptive", "multi_swap": "multi_flip",
    "multi_c1_flip": "single_refit", "m4_native": "m3_native",
    "multi_swap_no_gap": "multi_swap", "multi_flip_no_gap": "multi_flip",
    "tree_dynamic": "tree_original", "warm_05": "static_discovered", "warm_08": "static_discovered",
}
THRESHOLD = .15


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prior_report(values):
    return [dict(rank=j+1, pair=PAIRS[i].tolist(), prior=float(values[i]))
            for j, i in enumerate(np.argsort(-values, kind="stable"))]


def run_fit(args):
    start = perf_counter()
    models = c1.frozen_models()
    model = models["m3"]
    if not np.array_equal(model.pairs, PAIRS):
        raise ValueError("Frozen main checkpoint has unexpected pair ordering")
    main = Head(model.pair_prior, np.column_stack((model.lo, model.hi)),
                model.win_prior, model.size.prior)
    fit = generate("stable", FIT_SEED, args.fit_episodes)
    head, iterations = discover(fit, main)
    trees, folds = train_trees(fit, main, c1.subset)
    run = {}
    prior_only = replace(main, prior=head.prior)
    c1_head = SimpleNamespace(pairs=PAIRS, pair_prior=head.prior,
                              lo=head.rates[:, 0], hi=head.rates[:, 1], win_prior=head.default)
    timings = {}
    eval_base = SMOKE_SEED if args.smoke else EVAL_SEED
    eval_episodes = {}
    for k, scenario in enumerate(SCENARIOS):
        data = generate(scenario, eval_base + k, args.eval_episodes)
        if set(fit.episode_ids) & set(data.episode_ids) or data.targets is not None:
            raise ValueError("Evaluation scope overlaps fit or has a target")
        eval_episodes[scenario] = sorted(set(data.episode_ids))
        for tag in MASKS:
            split = masked(data, tag)
            names = {
                "static_main": (simple, main, {}),
                "static_discovered": (simple, prior_only, {}),
                "static_refit": (simple, head, {}),
                "warm_05": (simple, warm(main, head, .5), {}),
                "warm_08": (simple, warm(main, head, .8), {}),
                "single_refit": (simple, head, {"single": True}),
                "single_adaptive": (bocpd, head, {"swap": 0, "gaps": True, "single": True}),
                "multi_flip": (bocpd, head, {"swap": 0, "gaps": True}),
                "multi_swap": (bocpd, head, {"swap": .25, "gaps": True}),
                "multi_flip_no_gap": (bocpd, head, {"swap": 0, "gaps": False}),
                "multi_swap_no_gap": (bocpd, head, {"swap": .25, "gaps": False}),
            }
            key = f"{scenario}__{tag}"
            run[f"{key}__ids"] = np.asarray(split.query_ids)
            run[f"{key}__observed"] = split.X[:, :CUTOFF] >= 0
            timings[key] = {}
            for name, (method, arm_head, options) in names.items():
                begin = perf_counter()
                p, mass, trace = method(split, arm_head, **options)
                save_arm(run, key, name, c1.joint(p, arm_head.mix[split.context]), mass, trace)
                timings[key][name] = perf_counter() - begin

            begin = perf_counter()
            pw, pp, _, _, reset = c1.filter_win(c1_head, split, c1.Rule("polarity", 128), trace=True)
            save_arm(run, key, "multi_c1_flip", c1.joint(pw, head.mix[split.context]))
            run[f"{key}__multi_c1_flip__pair"] = pp
            run[f"{key}__multi_c1_flip__reset"] = reset[:, :CUTOFF]
            timings[key]["multi_c1_flip"] = perf_counter() - begin
            for native in ("m3", "m4"):
                begin = perf_counter()
                save_arm(run, key, f"{native}_native", models[native].predict(split))
                timings[key][f"{native}_native"] = perf_counter() - begin
            begin = perf_counter()
            inputs = feature_arrays(split, (CUTOFF,), ordering(head)[:6])[CUTOFF]
            for label, tree, features in zip(("tree_original", "tree_dynamic"), trees, inputs):
                save_arm(run, key, label, c1.joint(tree.predict_proba(features)[:, 1], head.mix[split.context]))
                timings[key][label] = perf_counter() - begin
            print(f"predicted {key}", flush=True)

    invariance = check_invariance(fit, head)
    prediction_file = args.out / "predictions.npz"
    np.savez_compressed(prediction_file, **run)
    paths = [Path(__file__), HERE / "fixtures.py", HERE / "filtering.py", HERE / "tree.py",
             C1, EXPERIMENTS / "research_next/synthetic.py",
             EXPERIMENTS / "05_adaptation_stress/stress.py", CHECKPOINT,
             HERE.parents[1] / "manifest.json", HERE.parents[1] / "coordinator/audit_run.py"]
    manifest = {
        "owner": "adapt_pairs", "card": "E5", "round": "main readiness",
        "season": "5fffdc4d0c700bbf61c3f911 (frozen main checkpoint only)",
        "fit_scope": "Independent stable synthetic pre-cutoff histories; four whole-episode cross-fit tree rankings. No cutoff targets created in fit-predict. Main carry comes from the frozen M3 checkpoint.",
        "evaluation_scope": "Independently generated prefix histories; cutoff law and hidden regime labels materialize in score only.",
        "fit_seed": FIT_SEED, "eval_base_seed": eval_base, "fit_episodes": args.fit_episodes,
        "eval_episodes": args.eval_episodes, "fit_episode_ids": sorted(set(fit.episode_ids)),
        "evaluation_episode_ids": eval_episodes, "rare_pairs": RARE, "scenarios": SCENARIOS,
        "cutoff": CUTOFF, "masks": MASKS, "hazard": HAZARD, "beam_budget": BUDGET,
        "segment_prior_strength": STRENGTH, "arms": [*names, "multi_c1_flip", "m3_native", "m4_native", "tree_original", "tree_dynamic"],
        "runtime_s": perf_counter() - start, "predict_timing_s": timings,
        "versions": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
                     "threadpools": threadpool_info()},
        "prediction_sha256": hash_file(prediction_file), "checks": invariance,
        "fit_command": sys.argv,
        "source_and_control_sha256": {str(p.relative_to(KIT)): hash_file(p) for p in paths},
    }
    json_write(args.out / "manifest.json", manifest)
    json_write(args.out / "pair_ranks.json", {
        "main": prior_report(main.prior), "eb_iterations": [prior_report(x) for x in iterations],
        "tree_folds": folds, "adaptation_rates": head.rates.tolist(),
        "adaptation_payout_mix": head.mix.tolist(),
    })


def save_arm(run, key, name, predictions, mass=None, trace=None):
    root = f"{key}__{name}"
    if np.any(~np.isfinite(predictions)) or np.any(predictions < 0) or not np.allclose(predictions.sum(axis=1), 1):
        raise ValueError(f"Invalid predictive vector {root}")
    run[root] = predictions
    if mass is not None:
        run[f"{root}__mass"] = mass
    if trace:
        for tag, values in trace.items():
            run[f"{root}__{tag}"] = values


def check_invariance(split, head):
    sample = c1.subset(split, np.array([1, 3, 5]))
    options = dict(swap=.25, gaps=True)
    original = bocpd(sample, head, **options)[0]
    reverse = bocpd(c1.subset(sample, np.array([2, 1, 0])), head, **options)[0]
    padded = bocpd(replace(sample, X=np.pad(sample.X, ((0, 0), (0, 7)), constant_values=-1)), head, **options)[0]
    assert np.array_equal(original, reverse[::-1])
    assert np.array_equal(original, padded)
    for tag, size in (("24+16", 40), ("24+8", 32)):
        assert np.all((masked(sample, tag).X >= 0).sum(axis=1) == size)
    assert np.all((masked(sample, "24+8").X < 0) | (masked(sample, "24+16").X >= 0))
    # At tick t, change the would-be target and subsequent outcomes.
    t = 160
    top = ordering(head)[:6]
    x = feature_arrays(sample, (t,), top)[t]
    h = sample.X.copy()
    h[:, t:] = (h[:, t:] + 3) % 5
    updated = feature_arrays(replace(sample, X=h), (t,), top)[t]
    assert all(np.array_equal(a, b) for a, b in zip(x, updated))
    return {"row_permutation": "passed", "post_cutoff_missing_padding": "passed",
            "nested_exact_mask_budgets": "passed", "tree_target_and_future_invariance": "passed"}


def first_delay(good, ticks, start, end):
    first = np.min(np.where(good & (ticks >= start) & (ticks < end), ticks - start, np.inf), axis=1)
    finite = np.isfinite(first)
    return {"detected_share": float(finite.mean()), "detected_n": int(finite.sum()),
            "median_ticks_detected": float(np.median(first[finite])) if finite.any() else None,
            "share_within_8_elapsed_ticks": float((first < 8).mean())}


def recovery(artifact, key, initial, alternate, direction, scenario, observed):
    mass = artifact[f"{key}__mass"].reshape(len(initial), len(PAIRS), 2)
    pair = mass.sum(axis=2)
    changes = SCENARIOS[scenario]
    final = alternate if scenario.startswith("swap") else initial
    chosen = pair.argmax(axis=1)
    pol = mass.sum(axis=1).argmax(axis=1)
    ticks = np.arange(CUTOFF)[None, :]
    reset = artifact[f"{key}__reset"]
    alarms = (reset >= THRESHOLD) & observed
    dropped = artifact[f"{key}__pruned"]
    trace = artifact[f"{key}__map"]
    readout = {
        "pair_top1": float((chosen == final).mean()), "polarity_top1": float((pol == direction).mean()),
        "true_pair_rank_mean": float((1 + (np.argsort(-pair, axis=1, kind="stable") != final[:, None]).cumprod(axis=1).sum(axis=1)).mean()),
        "true_pair_posterior_mean": float(pair[np.arange(len(final)), final].mean()),
        "pair_by_truth": {str(tuple(RARE[i])): {"n": int(sel.sum()),
                                               "top1": float((chosen[sel] == final[sel]).mean()) if sel.any() else None}
                          for i, rare in enumerate(RARE) for sel in [final == PAIR_ID[rare]]},
        "pruned_mean_per_tick": float(dropped.mean()), "pruned_p99": float(np.quantile(dropped, .99)),
        "pruned_max": float(dropped.max()), "pruned_mean_sum_per_episode_machine": float(dropped.sum(axis=1).mean()),
        "pruned_mean_missing_tick": float(dropped[~observed].mean()) if (~observed).any() else None,
        "pruned_mean_observed_tick": float(dropped[observed].mean()),
    }
    if f"{key}__mean_age" in artifact:
        readout["mean_age_at_last_history_tick"] = float(artifact[f"{key}__mean_age"][:, -1].mean())
    readout["alarm_before_first_change_share"] = float(np.any(
        alarms & (ticks >= 24) & (ticks < (changes[0] if changes else CUTOFF)), axis=1).mean())
    readout["changes"] = {}
    for j, change in enumerate(changes):
        end = changes[j+1] if j+1 < len(changes) else CUTOFF
        truth = 2 * (alternate if scenario.startswith("swap") else initial)
        truth += direction ^ int(scenario == "double_flip" and j == 0)
        good = trace == truth[:, None]
        eligible = good & observed
        readout["changes"][str(change)] = {
            "rule_recovery": first_delay(eligible, ticks, change, end),
            "reset_alarm": first_delay(alarms, ticks, change, end),
        }
    return readout


def run_score(args):
    manifest = json.loads((args.out / "manifest.json").read_text())
    prediction_file = args.out / "predictions.npz"
    if hash_file(prediction_file) != manifest["prediction_sha256"]:
        raise ValueError("Predictive artifact hash changed")
    rng = np.random.default_rng(7489)
    summary = {"scale": "raw sum of five squared probability errors, mean over machines",
               "alarm_rule": "posterior reset >= .15 on an observed tick",
               "rule_recovery": "first observed post-change tick with correct joint MAP; censor at next change or cutoff",
               "cases": {}}
    slices = []
    with np.load(prediction_file, allow_pickle=False) as data:
        for j, scenario in enumerate(SCENARIOS):
            split, truth, init, alt, direction = generate(scenario, manifest["eval_base_seed"] + j,
                                                           manifest["eval_episodes"], score=True)
            initial = np.array([PAIR_ID[tuple(p)] for p in init])
            alternate = np.array([PAIR_ID[tuple(p)] for p in alt])
            episode, group, counts = np.unique(split.episode_ids, return_inverse=True, return_counts=True)
            indices = rng.integers(len(episode), size=(4000, len(episode)))
            for tag in MASKS:
                key = f"{scenario}__{tag}"
                if not np.array_equal(data[f"{key}__ids"], split.query_ids):
                    raise ValueError("Forecast keys changed or out of order")
                observed = data[f"{key}__observed"]
                case = {}
                losses = {}
                for name in manifest["arms"]:
                    arm = f"{key}__{name}"
                    sq = (data[arm] - truth) ** 2
                    losses[name] = sq.sum(axis=1)
                    row = {"sse": float(sq.sum(axis=1).mean()), "p0_mse": float(sq[:, 0].mean()),
                           "sse_per_class": sq.mean(axis=0).tolist(),
                           "queries": split.n, "episodes": len(episode),
                           "observed_ticks": float(observed.sum(axis=1).mean())}
                    if f"{arm}__mass" in data:
                        row.update(recovery(data, arm, initial, alternate, direction, scenario, observed))
                    elif f"{arm}__pair" in data:
                        pair = data[f"{arm}__pair"]
                        final = alternate if scenario.startswith("swap") else initial
                        row["pair_top1"] = float((pair.argmax(axis=1) == final).mean())
                    case[name] = row
                for name, base in REFERENCES.items():
                    if name not in losses:
                        continue
                    diff = np.bincount(group, weights=losses[name] - losses[base]) / counts
                    case[name]["paired_minus"] = {"control": base, "sse_delta": float(diff.mean()),
                                                   "episode_ci95": np.quantile(diff[indices].mean(axis=1), [.025, .975]).tolist()}
                summary["cases"][key] = case
                for name, row in case.items():
                    ref = row.get("paired_minus", {})
                    slices.append({"scenario": scenario, "mask": tag, "arm": name, "queries": split.n,
                                   "episodes": len(episode), "observed": row["observed_ticks"], "SSE": row["sse"],
                                   "p0_MSE": row["p0_mse"], "pair_top1": row.get("pair_top1"),
                                   "control": ref.get("control"), "delta_SSE": ref.get("sse_delta"),
                                   "pruned_mean": row.get("pruned_mean_per_tick")})
    json_write(args.out / "synthetic_scores.json", summary)
    with (args.out / "slices.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(slices[0]))
        writer.writeheader()
        writer.writerows(slices)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("fit-predict", "score"))
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--fit-episodes", type=int, default=48)
    p.add_argument("--eval-episodes", type=int, default=48)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.stage == "fit-predict":
        args.out.mkdir(parents=True, exist_ok=True)
        if (args.out / "predictions.npz").exists():
            raise FileExistsError("Existing forecast is frozen; choose a fresh output directory.")
        run_fit(args)
    else:
        run_score(args)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
