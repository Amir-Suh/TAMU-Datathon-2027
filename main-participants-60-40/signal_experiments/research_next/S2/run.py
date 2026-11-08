"""Compare ways to handle unobserved lag bits under the same fixed win emissions."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import replace
import importlib.metadata
from itertools import chain, product
import json
from pathlib import Path
import platform
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import joblib
import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
KIT = HERE.parents[2]
EXPERIMENTS = KIT / "signal_experiments"
REPAIR = KIT / "m5_m6_repair"
FROZEN = REPAIR / "results/repaired-m5-m6"
PUBLIC = KIT / "public"
HYBRID = EXPERIMENTS / "combined/m3_match_split_win_pair_l31_factor_gru_size.csv"
DIAGNOSTICS = EXPERIMENTS / "ceiling_analysis/diagnostics.json"
sys.path[:0] = [str(KIT), str(REPAIR), str(EXPERIMENTS), str(HERE.parent)]

from models.common.crossfit import subset_split
from models.common.data import MISSING, PAYOUTS
from run_repaired_comparison import input_only
from score import compare, sha256
from slot_forecasting.contest import contest_score
from slot_forecasting.io import read_rows, read_submission, read_targets, write_json, write_rows, write_submission
from synthetic import generate, mask_history

AGES = (-1, 8, 16, 40, 80, 120)
BUDGETS = (0, 12, 24, 48, "all")
METHODS = ("drop", "expected", "exact")
SEED = 20261107
DRAWS = 10000


def native_library(directory):
    path = directory / "forward.so"
    if platform.system() == "Darwin":
        sdk = subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True).strip()
        compiler = ["xcrun", "--sdk", "macosx", "clang"]
        flags = ["-dynamiclib", "-isysroot", sdk]
    else:
        compiler, flags = ["cc"], ["-shared", "-fPIC"]
    subprocess.run([*compiler, "-std=c99", "-O3", *flags, str(HERE / "forward.c"), "-lm", "-o", str(path)],
                   check=True)
    library = ctypes.CDLL(str(path))
    array = np.ctypeslib.ndpointer
    double = array(dtype=np.float64, flags="C_CONTIGUOUS")
    library.exact_batch.argtypes = [array(dtype=np.int8, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int,
                                    array(dtype=np.int32, flags="C_CONTIGUOUS"), double, double, double,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, double, double]
    library.exact_batch.restype = None
    return library


def exact(native, history, cutoff, low, high, marginal, pair, polarity, age):
    ll = np.empty(len(cutoff))
    next_win = np.empty_like(ll)
    native.exact_batch(history, *history.shape, cutoff, low, high, marginal, *pair, polarity, age, ll, next_win)
    return ll, next_win


def shifted(history, lag):
    out = np.full_like(history, MISSING)
    out[:, lag:] = history[:, :-lag]
    return out


def pair_masks(history, cutoff, marginal, pair):
    """Return observed and independent-marginal pair contexts at each tick."""
    a, b = pair
    lag_a, lag_b = shifted(history, a), shifted(history, b)
    known = (lag_a != MISSING) & (lag_b != MISSING)
    difference = lag_a != lag_b
    p_a = np.where(lag_a == MISSING, marginal[:, None], lag_a)
    p_b = np.where(lag_b == MISSING, marginal[:, None], lag_b)
    expected_difference = p_a * (1 - p_b) + (1 - p_a) * p_b
    active = (np.arange(history.shape[1]) >= b)[None, :] & (cutoff[:, None] >= b)
    return known, difference, expected_difference, active


def bernoulli_log(history, probability):
    return np.where(history == 1, np.log(probability), np.log1p(-probability))


def approximate(history, cutoff, masks, low, high, marginal, polarity, age):
    known, difference, expected_difference, active = masks
    tick = np.arange(history.shape[1])
    direction = polarity ^ ((age >= 0) & (tick < cutoff[:, None] - age))
    aligned = difference == direction
    prob = np.where(aligned, high[:, None], low[:, None])
    probability = np.where(known & active, prob, marginal[:, None])
    observed = history != MISSING
    drop_ll = np.where(observed, bernoulli_log(history, probability), 0).sum(axis=1)

    high_weight = np.where(direction, expected_difference, 1 - expected_difference)
    weighted_log = high_weight * bernoulli_log(history, high[:, None])
    weighted_log += (1 - high_weight) * bernoulli_log(history, low[:, None])
    expected_ll = np.where(observed, np.where(active, weighted_log, bernoulli_log(history, marginal[:, None])),
                           0).sum(axis=1)

    # At the cutoff the current polarity applies, with independent Bernoulli
    # marginal imputation when only one or neither lag is observed.
    next_high_weight = np.where(polarity, expected_difference, 1 - expected_difference)
    next_mix = low[:, None] + next_high_weight * (high - low)[:, None]
    next_expected = np.where(active, next_mix, marginal[:, None])
    index = np.arange(len(history)), cutoff
    return (drop_ll, probability[index]), (expected_ll, next_expected[index])


def prefix_bits(split):
    ticks = np.arange(split.X.shape[1])
    return np.ascontiguousarray(np.where((split.X != MISSING) & (ticks < split.cutoff[:, None]),
                                         split.X > 0, MISSING), dtype=np.int8)


def predict(native, model, split):
    history = prefix_bits(split)
    cutoff = np.ascontiguousarray(split.cutoff, dtype=np.int32)
    low, high, marginal = (np.ascontiguousarray(rates[split.context])
                           for rates in (model.lo, model.hi, model.win_prior))
    null_ll = np.where(history != MISSING, bernoulli_log(history, marginal[:, None]), 0).sum(axis=1)
    hypotheses = tuple(product(model.pairs, (0, 1), AGES))
    likelihood = {key: np.empty((len(hypotheses), split.n)) for key in METHODS}
    wins = {key: np.empty_like(value) for key, value in likelihood.items()}
    j = 0
    for pair in model.pairs:
        masks = pair_masks(history, cutoff, marginal, pair)
        for polarity, age in product((0, 1), AGES):
            drop, expected = approximate(history, cutoff, masks, low, high, marginal, polarity, age)
            exact_result = exact(native, history, cutoff, low, high, marginal, pair, polarity, age)
            for method, result in zip(METHODS, (drop, expected, exact_result)):
                likelihood[method][j], wins[method][j] = result
            j += 1

    base_prior = (1 - model.null_weight) * np.asarray(model.pair_prior)[:, None, None, None] / 2
    ages = np.asarray(AGES)
    valid_ages = (cutoff[None, :] - ages[1:, None] >= model.min_segment)
    flip_mass = np.where(valid_ages.any(axis=0), model.p_flip, 0)
    flip_age_mass = valid_ages * flip_mass / np.maximum(valid_ages.sum(axis=0), 1)
    prior_shapes = {
        "constant": np.concatenate([np.ones((1, split.n)), np.zeros_like(flip_age_mass)], axis=0),
        "flip": np.concatenate([(1 - flip_mass)[None, :], flip_age_mass], axis=0),
    }
    complete = (history != MISSING).sum(axis=1) == cutoff
    for method in ("drop", "expected"):
        np.testing.assert_allclose(likelihood[method][:, complete], likelihood["exact"][:, complete],
                                   atol=5e-12, rtol=0)
        np.testing.assert_allclose(wins[method][:, complete], wins["exact"][:, complete], atol=1e-14, rtol=0)
    forecast = {}
    for scope, age_prior in prior_shapes.items():
        joint = np.broadcast_to(base_prior * age_prior[None, None, :, :],
                                (len(model.pairs), 2, len(AGES), split.n)).reshape(len(hypotheses), split.n)
        with np.errstate(divide="ignore"):
            prior = np.log(joint)
        for method in METHODS:
            log_joint = likelihood[method] + prior
            log_null = null_ll + np.log(model.null_weight)
            normalizer = np.logaddexp(logsumexp(log_joint, axis=0), log_null)
            posterior = np.exp(log_joint - normalizer)
            null = np.exp(log_null - normalizer)
            pair = posterior.reshape(len(model.pairs), 2, len(AGES), split.n).sum(axis=(1, 2))
            best = pair.argmax(axis=0)
            pair_index = np.asarray(model.pairs)[best]
            pair_index[null >= pair.max(axis=0)] = 0
            name = f"{method}_{scope}"
            forecast[name] = {"win": (posterior * wins[method]).sum(axis=0) + null * marginal,
                              "pair": pair_index, "null": null, "log_score": normalizer}
    return forecast


def brute_force(history, a, b, polarity, age, low, high, marginal):
    cutoff = len(history)
    flip = cutoff - age if age >= 0 else -1
    evidence = 0.0
    predicted = 0.0

    def p_win(bits, tick):
        if tick < b:
            return marginal
        direction = polarity ^ (flip >= 0 and tick < flip)
        return high if (bits[tick - a] != bits[tick - b]) == direction else low

    for bits in product((0, 1), repeat=cutoff):
        if any(value >= 0 and value != bit for value, bit in zip(history, bits)):
            continue
        weight = 1.0
        for tick, bit in enumerate(bits):
            p = p_win(bits, tick)
            weight *= p if bit else 1 - p
        evidence += weight
        predicted += weight * p_win(bits, cutoff)
    return np.log(evidence), predicted / evidence


def verify_native(native):
    rng = np.random.default_rng(SEED)
    verified = 0
    rates = tuple(np.asarray([r]) for r in (.13, .84, .42))
    for a, b in ((1, 2), (1, 3), (2, 4)):
        for polarity, age in product((0, 1), (-1, 3)):
            for row in (np.array([1, 0, 1, 0, 1, 1, 0, 1]), np.array([1, -1, -1, -1, 1, -1, 0, -1]),
                        np.full(8, -1), rng.integers(-1, 2, 8), np.array([-1, 1])):
                history = row.astype(np.int8)[None, :]
                cutoff = np.asarray([len(row)], dtype=np.int32)
                got = exact(native, history, cutoff, *rates, (a, b), polarity, age)
                wanted = brute_force(row, a, b, polarity, age, *(r[0] for r in rates))
                np.testing.assert_allclose([v[0] for v in got], wanted, atol=1e-12, rtol=0)
                verified += 1
    return verified


def combine(win, mix):
    return np.column_stack([1 - win, win[:, None] * mix])


def conditional(probability):
    return probability[:, 1:] / probability[:, 1:].sum(axis=1, keepdims=True)


def train_size_prior(train):
    """A fixed type-level payout mix for synthetic arms and synthetic reference."""
    win = (train.X > 0) & (np.arange(train.X.shape[1]) < train.cutoff[:, None])
    return np.asarray([np.bincount(train.X[win & (train.machine_type[:, None] == i)], minlength=5)[1:]
                       for i in range(3)], dtype=float)


def paired_interval(values, baseline):
    differences = values - baseline
    rng = np.random.default_rng(SEED)
    indices = rng.integers(len(values), size=(DRAWS, len(values)))
    return np.quantile(differences[indices].mean(axis=1), [.025, .975]).tolist()


def summarize_synthetic(result, fixture, model, size):
    split = fixture.split
    mix = size[split.machine_type]
    truth = split.targets
    baseline = combine(model.win_prior[split.context], mix)
    reference_sse = float(np.square(baseline - truth).sum(axis=1).mean())
    errors = {name: np.square(combine(value["win"], mix) - truth).sum(axis=1).reshape(-1, 8).mean(axis=1)
              for name, value in result.items()}
    summaries = {}
    for name, value in result.items():
        scope = name.split("_")[1]
        control = errors[f"drop_{scope}"]
        mse = float(errors[name].mean())
        r = mse / reference_sse
        summaries[name] = {
            "five_class_sse": mse, "win_mse": float(np.square(value["win"] - (1 - truth[:, 0])).mean()),
            "synthetic_R": r, "synthetic_points": 60 / (1 + r), "reference_sse": reference_sse,
            "pair_recovery": float((value["pair"] == fixture.pairs).all(axis=1).mean()),
            "null_selected": float((value["pair"] == 0).all(axis=1).mean()),
            "delta_sse_vs_drop": float((errors[name] - control).mean()),
            "delta_sse_vs_drop_95": paired_interval(errors[name], control),
        }
    return summaries


def verify_causal(native, model, split, expected):
    rng = np.random.default_rng(SEED)
    order = rng.permutation(split.n)
    cutoff = split.cutoff[order]
    reordered = subset_split(split, order)
    altered = np.pad(reordered.X, ((0, 0), (0, 8)), constant_values=MISSING)
    future = np.arange(altered.shape[1]) >= cutoff[:, None]
    altered[future] = rng.integers(0, len(PAYOUTS), size=future.sum())
    output = predict(native, model, replace(reordered, X=altered, targets=None, labels=None,
                                            reference=None, conditions=None))
    for name, value in expected.items():
        for key in value:
            np.testing.assert_allclose(value[key][order], output[name][key], atol=5e-12, rtol=0)


def public_mask_scores(out, native, model, data, split, mix, unmasked):
    # Freeze the full-prefix size mix across masks to isolate win-head sensitivity.
    by_budget = {"all": unmasked}
    last_observed = np.zeros_like(split.X, dtype=bool)
    ticks = np.arange(split.X.shape[1])
    recent = (ticks >= split.cutoff[:, None] - 40) & (ticks < split.cutoff[:, None])
    for budget in BUDGETS[:-1]:
        masked = mask_history(split, budget)
        observed = masked.X != MISSING
        assert not (last_observed & ~observed).any()
        np.testing.assert_array_equal(masked.X[recent], split.X[recent])
        last_observed = observed
        start = perf_counter()
        by_budget[str(budget)] = predict(native, model, masked)
        print(f"public validation older={budget:>3}: {perf_counter() - start:.2f}s", flush=True)
    truth = read_targets(data / "targets.csv")
    groups = {row["query_id"]: row["condition"] for row in
              read_rows(data / "conditions.csv", ("query_id", "condition"))}
    reference = read_submission(data / "reference.csv", split.query_ids)
    results = {}
    for budget in BUDGETS:
        results[str(budget)] = {}
        for name, value in by_budget[str(budget)].items():
            prediction = dict(zip(split.query_ids, combine(value["win"], mix)))
            scored = contest_score(prediction, truth, groups, reference, max_points=60)
            results[str(budget)][name] = {
                "R": scored["mean_loss_ratio"], "points": scored["technical_points"],
                "condition_R": {c: row["loss_ratio"] for c, row in scored["by_condition"].items()},
            }
    rows = ({"query_id": qid, "older_budget": budget,
             **{name: value["win"][i] for name, value in by_budget[str(budget)].items()}}
            for budget in BUDGETS for i, qid in enumerate(split.query_ids))
    first = next(rows)
    write_rows(out / "masked_public_win_predictions.csv", tuple(first), chain([first], rows))
    write_json(out / "masked_public_scores.json", results)


def main(out, native, smoke):
    out.mkdir(parents=True, exist_ok=True)
    clock = perf_counter()
    checks = {"brute_force_paths": verify_native(native)}
    checkpoint = FROZEN / "models.joblib"
    freeze = FROZEN / "freeze.json"
    checkpoint_sha = json.loads(freeze.read_text())["checkpoint_sha256"]
    diagnostic = json.loads(DIAGNOSTICS.read_text())
    assert sha256(checkpoint) == checkpoint_sha
    assert sha256(HYBRID) == diagnostic["forecast_sha256"]["best"]
    model = joblib.load(checkpoint)["m3"]
    train = input_only(PUBLIC, "train")
    size = train_size_prior(train)
    size /= size.sum(axis=1, keepdims=True)
    episodes = 12 if smoke else 24
    smoke_fixture = generate("stable", episodes=12, seed=SEED)
    before = perf_counter()
    predict(native, model, mask_history(smoke_fixture.split, 24))
    smoke_seconds = perf_counter() - before
    print(f"12-episode smoke took {smoke_seconds:.2f}s", flush=True)
    synthetic_scores = {}
    synthetic_rows = []
    for index, scenario in enumerate(("stable", "flip")):
        fixture = generate(scenario, episodes=episodes, seed=SEED + index, lead=16)
        synthetic_scores[scenario] = {}
        previous = np.zeros_like(fixture.split.X, dtype=bool)
        for budget in BUDGETS:
            split = fixture.split if budget == "all" else mask_history(fixture.split, budget)
            observed = split.X != MISSING
            assert not (previous & ~observed).any()
            assert (observed[:, -41:-1] == (fixture.split.X[:, -41:-1] != MISSING)).all()
            previous = observed
            start = perf_counter()
            result = predict(native, model, replace(split, labels=None, conditions=None, targets=None))
            elapsed = perf_counter() - start
            synthetic_scores[scenario][str(budget)] = {
                "episodes": episodes, "queries": split.n, "seconds": elapsed,
                "models": summarize_synthetic(result, fixture, model, size),
            }
            if budget == 24:
                verify_causal(native, model, subset_split(split, np.arange(12)),
                              {m: {k: v[:12] for k, v in r.items()} for m, r in result.items()})
            for i, query in enumerate(split.query_ids):
                for name, value in result.items():
                    synthetic_rows.append({
                        "scenario": scenario, "older_budget": budget, "query_id": query, "method": name,
                        "episode_id": split.episode_ids[i], "forecast_win": value["win"][i],
                        "true_win": 1 - fixture.split.targets[i, 0],
                        "lag_a": value["pair"][i, 0], "lag_b": value["pair"][i, 1],
                        "true_lag_a": fixture.pairs[i, 0], "true_lag_b": fixture.pairs[i, 1],
                        "null_probability": value["null"][i],
                    })
            print(f"synthetic {scenario:6} older={budget!s:>3}: {elapsed:.2f}s", flush=True)
    write_json(out / "synthetic_scores.json", synthetic_scores)
    write_rows(out / "synthetic_predictions.csv", tuple(synthetic_rows[0]), synthetic_rows)

    data = PUBLIC / "validation"
    validation = input_only(PUBLIC, "validation")
    result = predict(native, model, validation)
    verify_causal(native, model, subset_split(validation, np.arange(12)),
                  {m: {k: v[:12] for k, v in r.items()} for m, r in result.items()})
    hybrid = read_submission(HYBRID, validation.query_ids)
    mix = conditional(np.asarray([hybrid[q] for q in validation.query_ids]))
    paths = {}
    for name, value in result.items():
        path = out / ("validation_predictions.csv" if name == "exact_flip" else f"validation_{name}.csv")
        probability = combine(value["win"], mix)
        np.testing.assert_allclose(conditional(probability), mix, atol=5e-16, rtol=0)
        write_submission(path, dict(zip(validation.query_ids, probability)))
        paths[name] = path
    main_rows = ({"query_id": key, "method": name, "lag_a": value["pair"][i, 0], "lag_b": value["pair"][i, 1],
                  "null_probability": value["null"][i], "history_log_score": value["log_score"][i]}
                 for i, key in enumerate(validation.query_ids) for name, value in result.items())
    first = next(main_rows)
    write_rows(out / "validation_lag_posteriors.csv", tuple(first), chain([first], main_rows))
    scores = compare(data, HYBRID, paths, 60, DRAWS)
    contrasts = {scope: compare(data, paths[f"drop_{scope}"], {method: paths[f"{method}_{scope}"]
                      for method in ("expected", "exact")}, 60, DRAWS) for scope in ("constant", "flip")}
    write_json(out / "scores.json", scores)
    write_json(out / "matched_scores.json", contrasts)
    for name, value in scores["models"].items():
        print(f"main {name:18} {value['points']:.6f} ΔR={value['delta_R']:+.8f}", flush=True)
    public_mask_scores(out, native, model, data, validation, mix, result)
    metadata = json.loads((PUBLIC / "metadata.json").read_text())
    checks.update({"equal_complete_history_likelihood_and_forecasts": True,
                   "nested_masks_and_fixed_cutoffs": True, "permutation_padding_and_future_injection": True,
                   "hybrid_payout_mix_invariance": True})
    input_paths = [checkpoint, freeze, FROZEN / "plan.json", HYBRID, PUBLIC / "metadata.json", EXPERIMENTS / "folds.csv"]
    input_paths.extend(PUBLIC / "train" / p for p in ("queries.csv", "history.csv", "labels.csv"))
    input_paths.extend(data / p for p in ("queries.csv", "history.csv", "conditions.csv", "targets.csv", "reference.csv"))
    write_json(out / "plan.json", {
        "scope": "Train-only frozen M3 win emissions and priors. Main validation conditions, reference and "
                 "targets enter scoring after writing the predictions. Full forecasts retain the hybrid payout "
                 "head with transductive Fredy components.",
        "metadata": metadata, "payout_order": PAYOUTS,
        "runtime": {"python": platform.python_version(), "compiler": subprocess.check_output(
                    ["cc", "--version"], text=True).splitlines()[0], "numerical_threads": 2,
                    **{key: importlib.metadata.version(key) for key in
                       ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")}},
        "emissions": {"train_only_win_prior": model.win_prior.tolist(), "train_only_high": model.hi.tolist(),
                      "train_only_low": model.lo.tolist(), "pairs": model.pairs,
                      "train_only_pair_prior": model.pair_prior.tolist(), "null_weight": model.null_weight,
                      "train_only_flip_mass": model.p_flip, "one_flip_ages": AGES[1:]},
        "synthetic": {"scenarios": ["stable", "flip"], "seeds": [SEED, SEED + 1],
                      "episodes_per_scenario": episodes, "older_budgets": BUDGETS, "recent_ticks": 40,
                      "type_payout_mix": size.tolist()},
        "input_sha256": {str(p.relative_to(KIT)): sha256(p) for p in input_paths},
        "source_sha256": {str(p.relative_to(KIT)): sha256(p) for p in
                          [Path(__file__), HERE / "forward.c", HERE.parent / "synthetic.py",
                           EXPERIMENTS / "score.py", REPAIR / "run_repaired_comparison.py",
                           REPAIR / "models/m3_lag_pair_flip/model.py", KIT / "slot_forecasting/contest.py",
                           KIT / "slot_forecasting/scoring.py"]},
        "checks_passed": checks, "paired_bootstrap_draws": DRAWS, "paired_bootstrap_seed": SEED,
        "smoke_12_episodes_seconds": smoke_seconds, "elapsed_seconds": perf_counter() - clock,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--smoke", action="store_true", help="Use 12 synthetic episodes per scenario.")
    args = parser.parse_args()
    with TemporaryDirectory(prefix="s2-filter-") as temporary, threadpool_limits(limits=2):
        main(args.out.resolve(), native_library(Path(temporary)), args.smoke)
