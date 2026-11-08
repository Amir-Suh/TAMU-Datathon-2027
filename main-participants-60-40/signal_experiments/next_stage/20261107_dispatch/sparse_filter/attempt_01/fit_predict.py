"""Freeze the E4 fixed-emission and grouped support-gate forecasts."""
import __future__
import argparse
import csv
from dataclasses import replace
import gzip
from importlib.metadata import version
import importlib.util
from itertools import product
import json
from pathlib import Path
import sys
from types import ModuleType
from time import perf_counter

import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp, xlogy
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
RUN = HERE.parents[1]
EXP = RUN.parents[1]
KIT = EXP.parent
REPLAY = EXP / "01_fredy_replay"
ARCHIVE = HERE / "fredy_archive/src"
sys.path[:0] = [str(KIT), str(ARCHIVE), str(KIT / "m5_m6_repair"), str(EXP)]

# The archive is unchanged. Postpone its annotations when loading with the
# original repaired-model Python 3.9 runtime.
if sys.version_info < (3, 10):
    import tdx
    archived_data = ARCHIVE / "tdx/data.py"
    data_module = ModuleType("tdx.data")
    data_module.__file__ = str(archived_data)
    data_module.__package__ = "tdx"
    sys.modules["tdx.data"] = data_module
    exec(compile(archived_data.read_text(), str(archived_data), "exec",
                 flags=__future__.annotations.compiler_flag, dont_inherit=True), data_module.__dict__)
from tdx.data import Split as TreeSplit
from tdx.arms.gbdt.arm import GBDTArm
from tdx.arms.seq.arm import SeqArm
from models.common.crossfit import subset_split
from models.common.data import MISSING
from run_repaired_comparison import input_only
from score import sha256
from slot_forecasting.io import read_submission, write_json, write_rows, write_submission


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


S2 = load_module("s2_archive", EXP / "research_next/S2/run.py")
W1 = load_module("w1_archive", EXP / "research_next/W1/run.py")
SEED = S2.SEED
FOLDS = EXP / "folds.csv"
FROZEN = S2.FROZEN
CHECKPOINT = FROZEN / "models.joblib"
TRAIN_FIT = W1.ORIGINAL.parent / "train_fit.json"
DONORS = ("pair_l31", "gru128_factor_seed_1", "gru128_factor_seed_2", "gru128_factor_seed_3")
BUDGETS = ("released", "24plus16", "24plus8")
SHIFTS = (0, 8, 16, 24)
RIDGE = 0.0001
EPS = 1e-10


def eligible_inputs(split):
    seen = (np.arange(split.X.shape[1]) < split.cutoff[:, None]) & (split.X != MISSING)
    return replace(split, X=np.where(seen, split.X, MISSING).astype(np.int8),
                   labels=None, targets=None, conditions=None, reference=None)


def masks(split):
    # Shared old pool ends before the 16-tick suffix, so both masks retain the same old ticks.
    result = {budget: np.full_like(split.X, MISSING) for budget in BUDGETS[1:]}
    import hashlib
    for i, (query, cutoff) in enumerate(zip(split.query_ids, split.cutoff)):
        start = max(0, cutoff - 16)
        available = np.flatnonzero(split.X[i, :start] != MISSING)
        old = sorted(available, key=lambda t: hashlib.sha256(
            f"{SEED}:{query}:{cutoff}:{t}".encode()).digest())[:24]
        for budget, suffix in (("24plus16", 16), ("24plus8", 8)):
            result[budget][i, old] = split.X[i, old]
            start = max(0, cutoff - suffix)
            result[budget][i, start:cutoff] = split.X[i, start:cutoff]
    assert not ((result["24plus8"] != MISSING) & (result["24plus16"] == MISSING)).any()
    for matrix in result.values():
        observed = matrix != MISSING
        np.testing.assert_array_equal(matrix[observed], split.X[observed])
    return {"released": split, **{budget: replace(split, X=matrix) for budget, matrix in result.items()}}


def fixed_filter(native, model, split):
    """Use S2's native exact kernel and the same fixed Bernoulli law for the drop control."""
    history = S2.prefix_bits(split)
    cutoff = np.ascontiguousarray(split.cutoff, dtype=np.int32)
    low, high, marginal = (np.ascontiguousarray(r[split.context])
                           for r in (model.lo, model.hi, model.win_prior))
    ages = np.asarray(S2.AGES)
    eligible = cutoff[None, :] - ages[1:, None] >= model.min_segment
    mass = np.where(eligible.any(axis=0), model.p_flip, 0)
    age_prior = np.vstack((1 - mass, eligible * mass / np.maximum(eligible.sum(axis=0), 1)))
    prior = (1 - model.null_weight) * model.pair_prior[:, None, None, None] * age_prior[None, None, :, :] / 2
    with np.errstate(divide="ignore"):
        prior = np.log(prior)
    shape = (len(model.pairs), 2, len(ages), split.n)
    likelihood = {name: np.empty(shape) for name in ("drop", "forward")}
    wins = {name: np.empty(shape) for name in likelihood}
    support = np.empty((len(model.pairs), split.n))
    tick = np.arange(history.shape[1])
    observed = history != MISSING
    for index, pair in enumerate(model.pairs):
        lag_a, lag_b = (S2.shifted(history, lag) for lag in pair)
        complete = (lag_a != MISSING) & (lag_b != MISSING)
        support[index] = (complete & observed).sum(axis=1)
        different = lag_a != lag_b
        for polarity, age_index in product((0, 1), range(len(ages))):
            age = int(ages[age_index])
            direction = polarity ^ ((age >= 0) & (tick < cutoff[:, None] - age))
            probability = np.where(complete, np.where(different == direction, high[:, None], low[:, None]),
                                   marginal[:, None])
            k = (index, polarity, age_index)
            likelihood["drop"][k] = np.where(observed, S2.bernoulli_log(history, probability), 0).sum(axis=1)
            wins["drop"][k] = probability[np.arange(split.n), cutoff]
            likelihood["forward"][k], wins["forward"][k] = S2.exact(
                native, history, cutoff, low, high, marginal, pair, polarity, age)
    complete = observed.sum(axis=1) == cutoff
    np.testing.assert_allclose(likelihood["drop"][..., complete], likelihood["forward"][..., complete],
                               atol=5e-12, rtol=0)
    np.testing.assert_allclose(wins["drop"][..., complete], wins["forward"][..., complete], atol=1e-14, rtol=0)
    log_null = np.where(observed, S2.bernoulli_log(history, marginal[:, None]), 0).sum(axis=1)
    log_null += np.log(model.null_weight)
    result = {}
    for name in wins:
        log_joint = likelihood[name] + prior
        normalizer = np.logaddexp(logsumexp(log_joint, axis=(0, 1, 2)), log_null)
        posterior = np.exp(log_joint - normalizer)
        pair = posterior.sum(axis=(1, 2))
        null = np.exp(log_null - normalizer)
        win = (posterior * wins[name]).sum(axis=(0, 1, 2)) + null * marginal
        pair_norm = pair / np.maximum(pair.sum(axis=0), np.finfo(float).tiny)
        best = np.asarray(model.pairs)[pair.argmax(axis=0)].copy()
        best[null >= pair.max(axis=0)] = 0
        result[name] = dict(win=win, support=(pair_norm * support).sum(axis=0),
                            entropy=-xlogy(pair_norm, pair_norm).sum(axis=0),
                            pair=best, null=null, log_evidence=normalizer)
    return result


def donor_models():
    models = []
    for name in DONORS:
        directory = REPLAY / "checkpoints" / name
        if name == "pair_l31":
            import lightgbm
            meta = json.loads((directory / "features.json").read_text())
            tree = GBDTArm(**{**meta["params"], "threads": 1})
            tree.names = meta["names"]
            tree.keep = np.asarray(meta["keep"], dtype=bool)
            with gzip.open(directory / "model.txt.gz", "rt", encoding="utf-8") as source:
                tree.model = lightgbm.Booster(model_str=source.read())
            models.append(tree)
        else:
            models.append(SeqArm.load(directory))
    return models


def donor_predict(models, split):
    groups = {}
    for i, episode in enumerate(split.episode_ids):
        groups.setdefault(episode, []).append(i)
    index = np.asarray([sorted(indices, key=lambda i: split.machine_ids[i])
                        for indices in groups.values()])
    assert index.shape == (len(groups), 8)
    assert (split.cutoff[index] == split.cutoff[index[:, :1]]).all()
    y = split.X[index]
    # Keep tick positions identical to the archived Fredy inference contract.
    y = np.pad(y, ((0, 0), (0, 0), (0, max(0, 256-y.shape[-1]))), constant_values=MISSING)
    tree_split = TreeSplit(split.name, list(groups), y, split.machine_type[index], split.signal[index],
                           split.cutoff[index[:, 0]], (y >= 0).sum(axis=-1), np.asarray(split.query_ids)[index])
    components = np.asarray([model.predict(tree_split) for model in models])
    query_components = np.empty((len(models), split.n, 5))
    query_components[:, index.ravel()] = components.reshape(len(models), -1, 5)
    return (query_components[0] + query_components[1:].mean(axis=0)) / 2, query_components


def probability(path, split):
    mapping = read_submission(path)
    return np.asarray([mapping[q] for q in split.query_ids])


def fit_constant(data):
    base, donor, outcome, weights, _, _ = data.T
    delta = donor - base
    return float(np.clip(np.sum(weights * delta * (outcome - base))
                         / np.sum(weights * delta**2), 0, 1))


def fit_gate(data):
    constant = fit_constant(data)
    base, donor, outcome, weights, support, entropy = data.T
    # Optimization coordinates use support per 40 ticks and entropy in nats.
    design = np.column_stack((np.ones(len(data)), (40 - support) / 40, entropy))
    center = np.array([logit(np.clip(constant, EPS, 1-EPS)), 0, 0])
    delta = donor - base

    def objective(eta):
        gate = expit(design @ eta)
        error = base + gate * delta - outcome
        penalty = eta - center
        gradient = 2 * design.T @ (weights * error * delta * gate * (1 - gate))
        return np.dot(weights, error**2) + RIDGE * np.dot(penalty, penalty), gradient + 2*RIDGE*penalty

    fit = minimize(objective, center, method="L-BFGS-B", jac=True, bounds=[(-20, 20)]*3,
                   options={"ftol": 1e-13, "gtol": 1e-9})
    if not fit.success:
        raise RuntimeError(fit.message)
    return {"constant": constant, "lambda_max": 1., "eta": fit.x.tolist(),
            "eta_formula_raw": [fit.x[0], fit.x[1] / 40, fit.x[2]]}


def apply_gate(fit, support, entropy):
    eta = fit["eta_formula_raw"]
    return fit["lambda_max"] * expit(eta[0] + eta[1]*(40 - support) + eta[2]*entropy)


def gate_records(native, donors, train, out):
    with FOLDS.open(newline="") as source:
        folds = {row["episode_id"]: int(row["fold"]) for row in csv.DictReader(source) if row["split"] == "train"}
    assert set(folds) == set(train.episode_ids)
    row_folds = np.array([folds[e] for e in train.episode_ids])
    rows, info = [], []
    for fold in sorted(set(folds.values())):
        clock = perf_counter()
        fitted = subset_split(train, np.flatnonzero(row_folds != fold))
        held = subset_split(train, np.flatnonzero(row_folds == fold))
        refit = W1.refit_m3(fitted, tune=False)
        fold_info = {"fold": fold, "fit": sorted(set(fitted.episode_ids)), "held": sorted(set(held.episode_ids)),
                     "win_rates": refit.win_prior.tolist(), "high": refit.hi.tolist(), "low": refit.lo.tolist(),
                     "pair_prior": refit.pair_prior.tolist(), "flip_prior": float(refit.p_flip)}
        for shift in SHIFTS:
            frame = W1.prefix(held, shift)
            target = frame.labels.copy()
            causal = eligible_inputs(frame)
            for budget, masked in masks(causal).items():
                filtered = fixed_filter(native, refit, masked)["forward"]
                donor, _ = donor_predict(donors, masked)
                for i in np.flatnonzero(target != MISSING):
                    rows.append({"fold": fold, "episode_id": masked.episode_ids[i], "query_id": masked.query_ids[i],
                                 "shift": shift, "cutoff": int(masked.cutoff[i]), "budget": budget,
                                 "base": filtered["win"][i], "donor": 1 - donor[i, 0],
                                 "outcome": int(target[i] > 0), "support": filtered["support"][i],
                                 "entropy": filtered["entropy"][i]})
        fold_info["seconds"] = perf_counter() - clock
        info.append(fold_info)
        print(f"Fold {fold}: {len(fold_info['held'])} held episodes in {fold_info['seconds']:.2f}s", flush=True)
    write_rows(out / "train_grouped_gate_rows.csv", tuple(rows[0]), rows)
    data = np.asarray([[r[k] for k in ("base", "donor", "outcome", "support", "entropy")] for r in rows])
    episode = np.asarray([r["episode_id"] for r in rows])
    row_fold = np.asarray([r["fold"] for r in rows])

    def with_weights(selected):
        _, inverse, count = np.unique(episode[selected], return_counts=True, return_inverse=True)
        weight = 1 / (len(count)*count[inverse])
        return np.insert(data[selected], 3, weight, axis=1)

    diagnostics = {}
    for fold in sorted(set(row_fold)):
        fit = fit_gate(with_weights(row_fold != fold))
        base, donor, outcome, weights, support, entropy = with_weights(row_fold == fold).T
        diagnostic = {}
        for name, weight in (("constant", fit["constant"]), ("gated", apply_gate(fit, support, entropy))):
            win = base + weight * (donor - base)
            diagnostic[name] = {"grouped_binary_brier": float(np.dot(weights, (win-outcome)**2)),
                                "mean_weight": float(np.dot(weights, np.broadcast_to(weight, win.shape)))}
        diagnostics[int(fold)] = {"coefficients": fit, "held_loss": diagnostic}
    fit = fit_gate(with_weights(np.ones(len(data), dtype=bool)))
    write_json(out / "gate.json", {"fit": fit, "ridge": RIDGE, "train_rows": len(rows), "shifts": SHIFTS,
                                  "gate_only_held_fold_diagnostics": diagnostics,
                                  "note": "Shared Fredy donor retains original transductive exposure. Bayesian held-row emissions are fitted on the other folds. Outer gate diagnostics do not nest emission refits."})
    write_json(out / "episode_fold_manifest.json", {"source_sha256": sha256(FOLDS), "folds": info,
                                                     "episodes": sorted(folds), "cutoff_shifts": SHIFTS})
    return fit


def verify_causal(native, model, donors, split):
    rng = np.random.default_rng(SEED)
    index = rng.permutation(split.n)
    changed = subset_split(split, index)
    altered = np.pad(changed.X, ((0, 0), (0, 4)), constant_values=MISSING)
    future = np.arange(altered.shape[1]) >= changed.cutoff[:, None]
    altered[future] = rng.integers(0, 5, future.sum())
    changed = eligible_inputs(replace(changed, X=altered))
    original = masks(split)
    edited = masks(changed)
    for budget in BUDGETS:
        np.testing.assert_array_equal(original[budget].X[index], edited[budget].X[:, :split.X.shape[1]])
        base = fixed_filter(native, model, original[budget])
        new = fixed_filter(native, model, edited[budget])
        for arm in base:
            for key in base[arm]:
                np.testing.assert_allclose(base[arm][key][index], new[arm][key], atol=5e-12, rtol=0)
        np.testing.assert_allclose(donor_predict(donors, original[budget])[0][index],
                                   donor_predict(donors, edited[budget])[0], atol=1e-12, rtol=0)


def generate_synthetic(native, model, train, out):
    size = S2.train_size_prior(train)
    size /= size.sum(axis=1, keepdims=True)
    summaries = {}
    for scenario, seed in (("stable", SEED), ("flip", SEED + 1)):
        fixture = S2.generate(scenario, episodes=24, seed=seed, lead=16)
        summaries[scenario] = {}
        for budget, split in masks(eligible_inputs(fixture.split)).items():
            results = fixed_filter(native, model, split)
            summaries[scenario][budget] = S2.summarize_synthetic(
                {f"{name}_flip": info for name, info in results.items()}, fixture, model, size)
    write_json(out / "synthetic_scores.json", summaries)


def validate_donors(components, split):
    folder = "forecasts" if split.name == "validation" else "test_forecasts"
    errors = {}
    for name, predicted in zip(DONORS, components):
        archived = probability(REPLAY / folder / f"{name}.csv", split)
        errors[name] = float(np.max(abs(predicted - archived)))
        assert errors[name] < 1e-8, (name, errors[name])
    return errors


def predict_split(native, fixed, donors, split, fit, prior, out):
    name = split.name
    m3, m4 = fixed["m3"], fixed["m4"]
    hybrid_file = S2.HYBRID if name == "validation" else S2.HYBRID.with_stem(S2.HYBRID.stem + "_test")
    hybrid = probability(hybrid_file, split)
    frozen_mix = S2.conditional(hybrid)
    mask_rows, diagnostics, timing = [], [], {}
    forecasts = {}
    masked_by_budget = masks(split)
    donor_error = {}
    for budget in BUDGETS if name == "validation" else ("released",):
        clock = perf_counter()
        masked = masked_by_budget[budget]
        filter_out = fixed_filter(native, m3, masked)
        donor, components = donor_predict(donors, masked)
        mix = S2.conditional(donor)
        if budget == "released":
            donor_error = validate_donors(components, split)
            np.testing.assert_allclose(mix, frozen_mix, atol=1e-8, rtol=0)
            mix = frozen_mix
            if name == "validation":
                for arm, basename in (("drop", "validation_drop_flip"), ("forward", "validation_predictions")):
                    archived = probability(S2.HERE / f"{basename}.csv", split)
                    np.testing.assert_allclose(1-filter_out[arm]["win"], archived[:, 0], atol=5e-12, rtol=0)
        forward = filter_out["forward"]
        bayes_win = forward["win"]
        donor_win = 1 - donor[:, 0]
        gate = apply_gate(fit, forward["support"], forward["entropy"])
        champion_win = 1 - W1.w0.predict(m3, masked, prior)["candidate"][:, 0]
        if budget == "released":
            np.testing.assert_allclose(champion_win, 1-hybrid[:, 0], atol=5e-12, rtol=0)
        wins = {"drop": filter_out["drop"]["win"], "forward": bayes_win,
                "constant": bayes_win + fit["constant"]*(donor_win-bayes_win),
                "gated": bayes_win + gate*(donor_win-bayes_win),
                "hybrid": champion_win, "m3": m3.predict_win(masked), "m4": m4.predict_win(masked),
                "donor": donor_win}
        for tier, conditional in (("frozen_payout", frozen_mix), ("recomputed", mix)):
            for arm, win in wins.items():
                if budget == "released" and tier == "recomputed":
                    continue
                path = out / f"{name}_{budget}_{tier}_{arm}.csv"
                forecast = S2.combine(win, conditional)
                np.testing.assert_allclose(S2.conditional(forecast), conditional, atol=5e-16, rtol=0)
                write_submission(path, dict(zip(split.query_ids, forecast)))
                forecasts[f"{budget}/{tier}/{arm}"] = path.name
        for i, query in enumerate(split.query_ids):
            observed = np.flatnonzero(masked.X[i] != MISSING)
            mask_rows.append({"query_id": query, "episode_id": split.episode_ids[i], "budget": budget,
                              "cutoff": int(split.cutoff[i]), "observations": len(observed),
                              "ticks": " ".join(map(str, observed))})
            diagnostics.append({"query_id": query, "episode_id": split.episode_ids[i], "budget": budget,
                                "support": forward["support"][i], "entropy_nats": forward["entropy"][i],
                                "null": forward["null"][i], "lag_a": forward["pair"][i, 0],
                                "lag_b": forward["pair"][i, 1], "lambda_gated": gate[i],
                                "lambda_constant": fit["constant"], "donor_win": donor_win[i],
                                "delta_win_forward_vs_drop": bayes_win[i]-wins["drop"][i]})
        timing[budget] = perf_counter() - clock
        print(f"{name}/{budget}: {split.n} queries in {timing[budget]:.2f}s", flush=True)
    write_rows(out / f"{name}_mask_manifest.csv", tuple(mask_rows[0]), mask_rows)
    write_rows(out / f"{name}_support_diagnostics.csv", tuple(diagnostics[0]), diagnostics)
    return {"forecasts": forecasts, "timing": timing, "donor_max_error": donor_error}


def main(out, smoke):
    clock = perf_counter()
    dispatch = json.loads((RUN / "manifest.json").read_text())
    assert dispatch["round"] == "main"
    assert sha256(CHECKPOINT) == dispatch["input_sha256"][str(CHECKPOINT.relative_to(KIT))]
    fixed = joblib.load(CHECKPOINT)
    native = S2.native_library(out)
    checks = {"exhaustive_paths": S2.verify_native(native)}
    donors = donor_models()
    validation = input_only(KIT / "public", "validation")
    held = sorted(set(validation.episode_ids))[:12]
    preview = subset_split(validation, np.flatnonzero(np.isin(validation.episode_ids, held)))
    verify_causal(native, fixed["m3"], donors, preview)
    checks.update({"matched_complete_histories": True, "mask_nesting": True,
                   "mask_padding_permutation_and_future_invariance_12_episodes": True})
    if smoke:
        fit = {"constant": 0.3, "lambda_max": 1., "eta_formula_raw": [float(logit(.3)), 0., 0.]}
    else:
        train = W1.labeled_input(KIT / "public")
        fit = gate_records(native, donors, train, out)
    prior = np.asarray(json.loads(TRAIN_FIT.read_text())["learned_contrast_11_minus_00"])
    results = {"validation": predict_split(native, fixed, donors, preview if smoke else validation, fit, prior, out)}
    if not smoke:
        results["test"] = predict_split(native, fixed, donors, input_only(KIT / "public", "test"), fit, prior, out)
        generate_synthetic(native, fixed["m3"], train, out)
    write_json(out / "forecast_freeze.json", {
        "round": dispatch["round"], "season_id": dispatch["season_id"], "card": "E4", "owner": "sparse_filter",
        "smoke": smoke, "checks": checks, "result": results, "elapsed_seconds": perf_counter()-clock,
        "threads": 1, "packages": {n: version(n) for n in ("numpy", "scipy", "joblib", "lightgbm", "scikit-learn")},
        "fit_scope": {"drop_forward": "Frozen main train-only M3 emissions, matched S2 one-flip prior and grid.",
                      "gate": "New parameters fitted on main train sample outcomes using grouped train pseudo-cutoffs.",
                      "donor": "Frozen tree and three factored GRUs trained on all main splits' pre-cutoff histories. GRUs also saw main training endpoint labels.",
                      "validation": "Reused main development. Protected answers deferred to scoring.",
                      "masks": "Full released observation set or 24 hashed older observations before cutoff-16 plus 16/8 observed suffix ticks. Fixed cutoff and signal. Recompute Bayesian, tree and GRU inference. Frozen payout tier holds original hybrid payout only."},
        "input_sha256": {str(p.relative_to(KIT)): sha256(p) for p in (
            RUN / "manifest.json", CHECKPOINT, FROZEN / "freeze.json", TRAIN_FIT, FOLDS, S2.HYBRID,
            S2.HYBRID.with_stem(S2.HYBRID.stem + "_test"), S2.HERE / "run.py", S2.HERE / "forward.c",
            EXP / "research_next/W1/run.py", Path(__file__), *(p for n in DONORS for p in
            (REPLAY / "checkpoints" / n).iterdir()), *(KIT / "public" / split / item for split in
            ("train", "validation", "test") for item in ("history.csv", "queries.csv")),
            KIT / "public/train/labels.csv", *sorted(ARCHIVE.rglob("*.py")))},
        "output_sha256": {p.name: sha256(p) for p in sorted(out.iterdir()) if p.suffix in (".csv", ".json")},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        main(args.out.resolve(), args.smoke)
