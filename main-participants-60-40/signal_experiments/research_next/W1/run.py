"""Compare ordered lag-cell win refinements with frozen M3 posteriors."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
KIT = HERE.parents[2]
EXPERIMENTS = KIT / "signal_experiments"
REPAIR = KIT / "m5_m6_repair"
FROZEN = REPAIR / "results/repaired-m5-m6"
FOLDS = EXPERIMENTS / "folds.csv"
COMBINED = EXPERIMENTS / "combined/m3_match_split_win_pair_l31_factor_gru_size"
sys.path[:0] = [str(REPAIR), str(KIT), str(EXPERIMENTS)]

from models.common.components import combine, context_priors
from models.common.crossfit import subset_split
from models.common.data import MISSING, N_CONTEXTS, TYPES, Split, truncate
from models.common.selection import binary_logloss
from models.m3_lag_pair_flip.model import FlipLagPairModel, _lagged
from run_repaired_comparison import input_only
from score import compare, sha256
from slot_forecasting.io import read_submission, read_targets, require_ids, write_json, write_rows, write_submission
from slot_forecasting.schema import CONDITIONS

ORIGINAL = EXPERIMENTS / "04_four_lag_cells/run.py"
spec = importlib.util.spec_from_file_location("w0_match_split", ORIGINAL)
w0 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = w0
spec.loader.exec_module(w0)

CELLS = ((0, 3), (1, 2))  # 00/11, then 01/10
CELL_NAMES = ("00", "01", "10", "11")
STRENGTHS = (8, 20, 50)
SHIFTS = (0, 4, 8, 12, 16, 20, 24, 28)
PAIR_STRENGTH = 200.0
TYPE_STRENGTH = 200.0
SEED = 20261107
TARGET_PAIR = (4, 7)


@dataclass
class Contrasts:
    pooled: np.ndarray  # match/differ, current direction
    pair: np.ndarray  # pair, match/differ, current direction
    weighted_differences: np.ndarray  # machine type, pair, match/differ, current direction
    effective_counts: np.ndarray  # same shape as weighted_differences

    def typed(self):
        return ((self.weighted_differences + TYPE_STRENGTH * self.pair)
                / (self.effective_counts + TYPE_STRENGTH))


def ordered_counts(w, cutoff, pair, tau_grid):
    """Post-flip complete triples in cell order 00, 01, 10, 11."""
    a, b = pair
    lag_a, lag_b = _lagged(w, a), _lagged(w, b)
    available = (w >= 0) & (lag_a >= 0) & (lag_b >= 0)
    codes = 2 * lag_a + lag_b
    grid = np.r_[0, np.minimum(tau_grid, w.shape[1])]
    suffixes = []
    for cell in range(4):
        valid = available & (codes == cell)
        counts = []
        for mask in (valid, valid & (w == 1)):
            prefix = np.column_stack((np.zeros(w.shape[0], dtype=int), mask.cumsum(axis=1)))
            counts.append(prefix[:, -1:] - prefix[:, grid])
        suffixes.append(counts)
    n, k = np.array(suffixes).transpose(1, 2, 3, 0)
    rows = np.arange(len(cutoff))
    current = np.where((lag_a[rows, cutoff] >= 0) & (lag_b[rows, cutoff] >= 0),
                       codes[rows, cutoff], MISSING)
    return n, k, current


def cell_evidence(n, k, children):
    left, right = children
    total = n[..., left] + n[..., right]
    effective = n[..., left] * n[..., right] / np.maximum(total, 1)
    observed = k[..., right] / np.maximum(n[..., right], 1) - k[..., left] / np.maximum(n[..., left], 1)
    fraction_right = (n[..., right] + 1) / (total + 2)
    return total, effective, observed, fraction_right


def train_contrasts(model, train):
    w = w0.win_history(train)
    posterior = model._infer(train)["ind"]["post"]
    shape = (len(TYPES), len(model.pairs), len(CELLS), 2)
    effective = np.zeros(shape)
    weighted = np.zeros(shape)
    for i, pair in enumerate(model.pairs):
        n, k, _ = ordered_counts(w, train.cutoff, pair, model.tau_grid)
        for group, children in enumerate(CELLS):
            _, strength, rate_diff, _ = cell_evidence(n, k, children)
            weights = posterior[:, i] * strength[:, None, :]
            for machine_type in range(len(TYPES)):
                rows = train.machine_type == machine_type
                effective[machine_type, i, group] = weights[rows].sum(axis=(0, 2))
                weighted[machine_type, i, group] = (weights[rows] * rate_diff[rows, None, :]).sum(axis=(0, 2))
    pooled = weighted.sum(axis=(0, 1)) / (effective.sum(axis=(0, 1)) + w0.GLOBAL_PRIOR)
    pair = (weighted.sum(axis=0) + PAIR_STRENGTH * pooled) / (effective.sum(axis=0) + PAIR_STRENGTH)
    return Contrasts(pooled, pair, weighted, effective)


def centered_children(parent, fraction_right, contrast):
    fraction_left = 1 - fraction_right
    lower = np.maximum(-(1 - parent) / fraction_right, -parent / fraction_left)
    upper = np.minimum(parent / fraction_right, (1 - parent) / fraction_left)
    contrast = np.clip(contrast, lower, upper)
    return parent - fraction_right * contrast, parent + fraction_left * contrast


def forecast(model, split, fit, variants):
    """Evaluate predeclared arms with one shared upstream posterior."""
    w = w0.win_history(split)
    inference = model._infer(split)
    posterior = inference["ind"]["post"]
    pair_posterior = posterior.sum(axis=(2, 3))
    top = pair_posterior.argmax(axis=1)
    delta = {name: np.zeros(split.n) for name in variants}
    baseline_delta = np.zeros(split.n)
    top_context = np.full(split.n, MISSING)
    top_support = np.zeros((split.n, 4))
    weighted_counts = np.zeros((split.n, 4))
    eligible_mass = np.zeros((split.n, 4))
    max_center_error = 0.0
    typed = fit.typed() if any(arm == "W1c" for arm, _ in variants.values()) else None
    for i, pair in enumerate(model.pairs):
        n, k, current = ordered_counts(w, split.cutoff, pair, model.tau_grid)
        top_context[top == i] = current[top == i]
        top_support[top == i] = n[top == i, 0]
        parent = inference["core"]["pred"][:, i]
        post = posterior[:, i]
        for group, children in enumerate(CELLS):
            use = np.isin(current, children)
            if not np.any(use):
                continue
            total, effective, observed, frac_right = cell_evidence(n[use], k[use], children)
            parent_group, post_group = parent[use], post[use]
            frac_right = frac_right[:, None, :]
            local = (effective * observed)[:, None, :]
            support = (total / (total + w0.MIN_HISTORY_PSEUDO))[:, None, :]
            supported = effective[:, None, :] > 0
            right = (current[use] == children[1])[:, None, None]
            for cell in children:
                at_cell = use & (current == cell)
                weighted_counts[at_cell, cell] += (post[at_cell] * n[at_cell, None, :, cell]).sum(axis=(1, 2))
                eligible_mass[at_cell, cell] += (post_group[current[use] == cell]
                                                * supported[current[use] == cell]).sum(axis=(1, 2))

            def correction(contrast):
                nonlocal max_center_error
                left_p, right_p = centered_children(parent_group, frac_right, contrast)
                centered = (1 - frac_right) * left_p + frac_right * right_p
                max_center_error = max(max_center_error, float(np.abs(centered - parent_group).max()))
                return (post_group * (np.where(right, right_p, left_p) - parent_group)).sum(axis=(1, 2))

            pooled = fit.pooled[group, None, :, None]
            if group == 0:
                # Preserve the published W0 match refinement, including its one-child fallback.
                base_contrast = (local + w0.LOCAL_PRIOR * pooled) / (effective[:, None, :] + w0.LOCAL_PRIOR) * support
                base_change = correction(base_contrast)
                baseline_delta[use] += base_change
            for name, (arm, strength) in variants.items():
                if group == 0 and arm == "W1a":
                    delta[name][use] += base_change
                    continue
                prior = pooled
                if arm == "W1b":
                    prior = fit.pair[i, group, None, :, None]
                if arm == "W1c":
                    prior = typed[split.machine_type[use], i, group, :, None]
                weight = strength / (effective[:, None, :] + strength) * support
                if group == 0:
                    contrast = base_contrast + np.where(supported, weight * (prior - pooled), 0.0)
                else:
                    contrast = np.where(supported, local / (effective[:, None, :] + strength) * support
                                        + weight * prior, 0.0)
                delta[name][use] += correction(contrast)
    frozen = inference["p_win"]
    wins = {"M3": frozen, "W0": frozen + baseline_delta}
    wins.update({name: frozen + change for name, change in delta.items()})
    if any(np.any(~np.isfinite(p) | (p < 0) | (p > 1)) for p in wins.values()):
        raise ValueError("Invalid win probability.")
    if not np.allclose(posterior.sum(axis=(1, 2, 3)) + inference["ind"]["post0"], 1, atol=1e-12, rtol=0):
        raise ValueError("Frozen posterior failed normalization.")
    return wins, {"top_pair": top, "top_context": top_context, "top_no_flip_cell_counts": top_support,
                  "posterior_weighted_current_cell_count": weighted_counts,
                  "supported_posterior_mass": eligible_mass,
                  "max_center_error": max_center_error}


def prefix(split, shift):
    """An earlier pseudo cutoff has no contemporaneous signal in the release."""
    shortened = truncate(split, shift)
    if shift:
        shortened = replace(shortened, signal=np.zeros(shortened.n, dtype=int))
    return shortened


def refit_m3(train, *, tune):
    """Fit every upstream win parameter using only the supplied fit episodes."""
    model = FlipLagPairModel()
    model.win_prior, _ = context_priors(train)
    model.hi, model.lo = np.full(N_CONTEXTS, 0.85), np.full(N_CONTEXTS, 0.08)
    model.pair_prior = np.full(len(model.pairs), 1 / len(model.pairs))
    model.tau_q = np.full(len(model.tau_grid), 1 / len(model.tau_grid))
    for _ in range(model.em_iterations):
        fitted = model._infer(train, model.default_kappa, store_counts=True)
        model._update_priors(fitted, train)
        del fitted
    model.params = {"kappa": model.default_kappa}
    if tune:
        frames = [prefix(train, shift) for shift in SHIFTS]
        model.selection = []
        for item in model.GRID:
            kappa = item["kappa"]
            p = np.concatenate([model._infer(s, kappa)["p_win"] for s in frames])
            y = np.concatenate([s.labels for s in frames])
            loss = binary_logloss(p, y > 0, y != MISSING)
            model.selection.append({"kappa": kappa, "train_logloss": loss})
        model.params = {"kappa": min(model.selection, key=lambda result: result["train_logloss"])["kappa"]}
    return model


def labeled_input(data):
    train = input_only(data, "train")
    with (data / "train/labels.csv").open(newline="") as stream:
        labels = {r["query_id"]: int(r["outcome"]) for r in csv.DictReader(stream)}
    require_ids(labels, train.query_ids)
    from models.common.data import PAYOUT_INDEX
    return replace(train, labels=np.array([PAYOUT_INDEX[labels[q]] for q in train.query_ids], dtype=np.int8))


def grouped_metrics(episode, y, p, mask):
    episode = episode[mask]
    if not len(episode):
        return {"episodes": 0, "rows": 0}
    _, inverse, counts = np.unique(episode, return_inverse=True, return_counts=True)
    p, y = p[mask], y[mask]
    brier = (p - y) ** 2
    clipped = np.clip(p, 1e-9, 1 - 1e-9)
    logloss = -np.where(y, np.log(clipped), np.log1p(-clipped))
    return {"episodes": len(counts), "rows": len(y),
            "binary_brier": float(np.mean(np.bincount(inverse, weights=brier) / counts)),
            "binary_logloss": float(np.mean(np.bincount(inverse, weights=logloss) / counts))}


def select_strength(train):
    with FOLDS.open(newline="") as stream:
        fold_by_episode = {r["episode_id"]: int(r["fold"]) for r in csv.DictReader(stream) if r["split"] == "train"}
    require_ids(fold_by_episode, set(train.episode_ids))
    row_folds = np.array([fold_by_episode[e] for e in train.episode_ids])
    variants = {f"{arm}_{strength}": (arm, strength) for arm in ("W1a", "W1b")
                for strength in STRENGTHS}
    rows, fold_info, refits = [], [], []
    for fold in sorted(set(row_folds)):
        fit = subset_split(train, np.flatnonzero(row_folds != fold))
        held = subset_split(train, np.flatnonzero(row_folds == fold))
        fit_episodes, held_episodes = sorted(set(fit.episode_ids)), sorted(set(held.episode_ids))
        if set(fit_episodes) & set(held_episodes):
            raise ValueError("Fold episode overlap.")
        print(f"Refitting M3 for train fold {fold}, {len(fit_episodes)} fit episodes", flush=True)
        model = refit_m3(fit, tune=True)
        contrast = train_contrasts(model, fit)
        refits.append((held, model, contrast))
        fold_info.append({"fold": int(fold), "fit_episodes": len(fit_episodes), "held_episodes": len(held_episodes),
                          "fit_episode_sha256": id_hash(fit_episodes), "held_episode_sha256": id_hash(held_episodes),
                          "m3_kappa": model.params["kappa"], "m3_kappa_selection": model.selection,
                          "pooled_contrasts": contrast.pooled.tolist()})
        for shift in SHIFTS:
            frame = prefix(held, shift)
            p, details = forecast(model, frame, contrast, variants)
            top_is_47 = details["top_pair"] == model.pairs.index(TARGET_PAIR)
            for i in np.flatnonzero(frame.labels != MISSING):
                rows.append({"fold": int(fold), "query_id": frame.query_ids[i], "episode_id": frame.episode_ids[i],
                             "shift": shift, "top_pair_4_7": bool(top_is_47[i]), "win": int(frame.labels[i] > 0),
                             **{name: v[i] for name, v in p.items()}})
    episode = np.array([r["episode_id"] for r in rows])
    y = np.array([r["win"] for r in rows], dtype=bool)
    top_47 = np.array([r["top_pair_4_7"] for r in rows])
    endpoints = np.array([r["shift"] == 0 for r in rows])
    masks = {"all": np.ones(len(rows), dtype=bool), "top_pair_4_7": top_47,
             "actual_cutoff": endpoints, "earlier_cutoffs": ~endpoints}
    predictions = {name: np.array([r[name] for r in rows]) for name in ("M3", "W0", *variants)}
    metrics = {name: {label: grouped_metrics(episode, y, p, mask) for label, mask in masks.items()}
               for name, p in predictions.items()}
    winner = min(STRENGTHS, key=lambda strength: metrics[f"W1a_{strength}"]["all"]["binary_brier"])
    baseline, pooled, pair = "W0", f"W1a_{winner}", f"W1b_{winner}"

    def improves(candidate, control):
        return all(metrics[candidate][label]["binary_brier"] < metrics[control][label]["binary_brier"]
                   for label in ("all", "top_pair_4_7"))

    type_gate = improves(pair, pooled)
    if type_gate:
        name = f"W1c_{winner}"
        values = []
        for held, model, contrast in refits:
            for shift in SHIFTS:
                frame = prefix(held, shift)
                p, _ = forecast(model, frame, contrast, {name: ("W1c", winner)})
                values.extend(p[name][frame.labels != MISSING])
        for row, p in zip(rows, values, strict=True):
            row[name] = p
        metrics[name] = {label: grouped_metrics(episode, y, np.array(values), mask)
                         for label, mask in masks.items()}
    shortlist = [pooled, pair]
    if type_gate and improves(f"W1c_{winner}", pair):
        shortlist.append(f"W1c_{winner}")
    # Keep the lowest overall Brier among arms that improve both required slices.
    eligible = [name for name in shortlist if improves(name, baseline)]
    best = min(eligible, key=lambda name: metrics[name]["all"]["binary_brier"]) if eligible else baseline
    return winner, type_gate, best.split("_")[0], rows, {
        "folds": fold_info, "metrics": metrics, "selected_strength": winner,
        "selection": "Lowest episode-averaged Bernoulli Brier for W1a across all held prequential rows.",
        "type_gate": type_gate, "stage_gate": best != baseline, "train_selected_win_head": best.split("_")[0],
        "gate": "Strictly lower episode-averaged Brier overall and in frozen top-pair (4,7).",
    }


def id_hash(ids):
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def make_synthetic(rng, law, name, episodes):
    """Stationary order-eight process with independent payouts conditional on a win."""
    count, cutoff = episodes * 8, 236
    true_pairs = np.array([(2, 5), (4, 7), (5, 8)])
    pairs = true_pairs[np.arange(count) % len(true_pairs)]
    direction = np.arange(count) % 2
    pair_id = np.arange(count) % len(true_pairs)
    match_high = np.where(direction == 0, 0.74, 0.26)
    differ_high = 1 - match_high
    rates = np.column_stack((match_high, differ_high, differ_high, match_high))
    if law != "parity":
        match_offset = 0.08
        differ_offset = 0.16
        if law == "pair_specific":
            match_offset = np.array([0.06, -0.17, 0.11])[pair_id]
            differ_offset = np.array([0.10, -0.19, 0.15])[pair_id]
        rates += np.column_stack((-np.broadcast_to(match_offset, count), -np.broadcast_to(differ_offset, count),
                                  np.broadcast_to(differ_offset, count), np.broadcast_to(match_offset, count)))
    draw = rng.random((count, cutoff + 1))
    binary = np.zeros((count, cutoff + 1), dtype=np.int8)
    binary[:, :8] = draw[:, :8] < 0.5
    rows = np.arange(count)
    for tick in range(8, cutoff + 1):
        cell = 2 * binary[rows, tick - pairs[:, 0]] + binary[rows, tick - pairs[:, 1]]
        p_win = rates[rows, cell]
        binary[:, tick] = draw[:, tick] < p_win
    size = np.array([0.64, 0.22, 0.11, 0.03])
    payout = np.where(binary, rng.choice(np.arange(1, 5), binary.shape, p=size), 0).astype(np.int8)
    labels = payout[:, -1].copy()
    payout[:, -1] = MISSING
    # Some episodes use a nested sparse history before a dense 40-tick suffix.
    mask_rows = np.flatnonzero((rows // 8) % 4 == 0)
    mask = rng.random((len(mask_rows), cutoff - 40)) < 0.85
    payout[mask_rows[:, None], np.arange(cutoff - 40)[None, :]] = np.where(
        mask, MISSING, payout[mask_rows[:, None], np.arange(cutoff - 40)[None, :]])
    split = Split(name, [f"{name}-{i}" for i in rows], np.array([f"{name}-e{i // 8}" for i in rows]),
                  [f"m{i % 8}" for i in rows], rows % len(TYPES), np.zeros(count, dtype=int),
                  np.full(count, cutoff), payout, labels)
    return split, pairs, p_win, size


def synthetic_checks(strength, type_gate):
    variant = {name: (name, strength) for name in ("W1a", "W1b")}
    if type_gate:
        variant["W1c"] = ("W1c", strength)
    results = {}
    for law in ("parity", "ordered", "pair_specific"):
        rng = np.random.default_rng(SEED)
        train, _, _, _ = make_synthetic(rng, law, "synthetic-fit", 72)
        held, true_pairs, true_win, size = make_synthetic(rng, law, "synthetic-held", 48)
        model = refit_m3(train, tune=False)
        fit = train_contrasts(model, train)
        forecasts, details = forecast(model, held, fit, variant)
        true_forecasts = combine(true_win, np.broadcast_to(size, (held.n, len(size))))
        result = {name: {"p0_mse": float(np.mean((p - true_win) ** 2)),
                         "five_class_sse": float(np.mean(np.square(
                             combine(p, np.broadcast_to(size, (held.n, len(size)))) - true_forecasts).sum(axis=1))),
                         "mean_abs_win_delta_vs_W0": float(np.mean(np.abs(p - forecasts["W0"])))}
                  for name, p in forecasts.items()}
        result["true_pair_recovery"] = float(np.all(np.array(model.pairs)[details["top_pair"]] == true_pairs, axis=1).mean())
        result["pooled_contrasts"] = fit.pooled.tolist()
        result["pair_contrasts"] = {f"{a}-{b}": fit.pair[model.pairs.index((a, b))].tolist()
                                    for a, b in ((2, 5), TARGET_PAIR, (5, 8))}
        results[law] = result
    return {"fit_episodes_per_law": 72, "held_episodes_per_law": 48, "seed": SEED,
            "support": "25% of episodes randomly mask 85% of history preceding the last 40 ticks",
            "mechanism_checks": {"pooled_beats_W0_on_ordered": results["ordered"]["W1a"]["p0_mse"] < results["ordered"]["W0"]["p0_mse"],
                                 "pair_beats_pooled_on_pair_specific": results["pair_specific"]["W1b"]["p0_mse"] < results["pair_specific"]["W1a"]["p0_mse"],
                                 "parity_pooled_differ_contrast_below_0_04": bool(np.max(np.abs(results["parity"]["pooled_contrasts"][1])) < 0.04)},
            "laws": results}


def causal_checks(model, split, fit, variants):
    sample = subset_split(split, np.arange(16))
    expected, diagnostics = forecast(model, sample, fit, variants)
    reversed_rows = np.arange(sample.n)[::-1]
    reordered, _ = forecast(model, subset_split(sample, reversed_rows), fit, variants)
    padded, _ = forecast(model, replace(sample, X=np.pad(sample.X, ((0, 0), (0, 13)), constant_values=MISSING)),
                         fit, variants)
    for arm in expected:
        np.testing.assert_array_equal(expected[arm], reordered[arm][reversed_rows])
        np.testing.assert_array_equal(expected[arm], padded[arm])
    missing = replace(sample, X=np.full_like(sample.X, MISSING))
    empty, _ = forecast(model, missing, fit, variants)
    for arm in empty:
        np.testing.assert_array_equal(empty[arm], empty["M3"])
    # Three isolated observations provide some current contexts but at most one
    # complete historical triple. Every new contrast therefore lacks one child.
    isolated = np.full_like(sample.X, MISSING)
    for offset, value in ((10, 0), (7, 0), (4, 1)):
        isolated[np.arange(sample.n), sample.cutoff - offset] = value
    unsupported, _ = forecast(model, replace(sample, X=isolated), fit, variants)
    for arm in variants:
        np.testing.assert_array_equal(unsupported[arm], unsupported["W0"])
    parent = np.array([0.04, 0.42, 0.97])
    children = centered_children(parent, np.array([0.12, 0.6, 0.87]), np.zeros(3))
    for child in children:
        np.testing.assert_array_equal(parent, child)
    w = w0.win_history(sample)
    checks = 0
    for pair in model.pairs[::7]:
        a, b = pair
        grid = model.tau_grid[::13]
        n, k, _ = ordered_counts(w, sample.cutoff, pair, grid)
        for row, cutoff in enumerate(sample.cutoff):
            for j, tau in enumerate(np.r_[0, grid]):
                for cell in range(4):
                    kept = [t for t in range(max(b, tau), cutoff)
                            if w[row, t] >= 0 and w[row, t - a] == cell // 2 and w[row, t - b] == cell % 2]
                    assert n[row, j, cell] == len(kept) and k[row, j, cell] == w[row, kept].sum()
                    checks += 1

    old_cutoff = sample.cutoff - 16
    after_earlier = np.arange(sample.X.shape[1])[None, :] >= old_cutoff[:, None]
    complete = sample.X.copy()
    complete[after_earlier] = 4
    earlier = prefix(replace(sample, labels=np.zeros(sample.n, dtype=int)), 16)
    perturbed = prefix(replace(sample, X=complete, labels=np.ones(sample.n, dtype=int)), 16)
    early, _ = forecast(model, earlier, fit, variants)
    early_perturbed, _ = forecast(model, perturbed, fit, variants)
    for arm in early:
        np.testing.assert_array_equal(early[arm], early_perturbed[arm])
    cutoff_outcome_changed = replace(sample, labels=np.ones(sample.n, dtype=int))
    ignored, _ = forecast(model, cutoff_outcome_changed, fit, variants)
    for arm in expected:
        np.testing.assert_array_equal(expected[arm], ignored[arm])
    invalid = replace(sample, X=sample.X.copy())
    invalid.X[0, invalid.cutoff[0]] = 1
    try:
        forecast(model, invalid, fit, variants)
    except ValueError as error:
        if str(error) != "Found history at or after cutoff.":
            raise
    else:
        raise ValueError("Accepted an at-cutoff payout as a history row.")
    return {"brute_force_strict_pre_cutoff_count_checks": checks, "identical_with_row_reordering": True,
            "identical_with_missing_padding": True, "zero_contrast_returns_exact_parent": True,
            "missing_all_history_returns_exact_parent": True, "missing_one_child_collapses_to_W0": True,
            "cutoff_label_ignored": True,
            "at_cutoff_input_rejected": True, "earlier_forecast_ignores_perturbed_at_cutoff_and_future": True,
            "max_weighted_child_center_error": diagnostics["max_center_error"]}


def array_at(path, split):
    records = read_submission(path, split.query_ids)
    return np.array([records[q] for q in split.query_ids])


def conditional_mix(p):
    return p[:, 1:] / p[:, 1:].sum(axis=1, keepdims=True)


def compose_forecasts(model, split, fit, variants, hybrid_path, outputs):
    p, details = forecast(model, split, fit, variants)
    frozen = array_at(hybrid_path, split)
    donor = conditional_mix(frozen)
    match_replay = combine(p["W0"], donor)
    np.testing.assert_allclose(match_replay, frozen, atol=1e-12, rtol=0)
    checks = {"W0_replay_max_abs_delta": float(np.abs(match_replay - frozen).max())}
    for arm, path in outputs.items():
        probabilities = combine(p[arm], donor)
        write_submission(path, dict(zip(split.query_ids, probabilities)))
        actual = array_at(path, split)
        np.testing.assert_allclose(probabilities, actual, atol=1e-12, rtol=0)
        np.testing.assert_allclose(conditional_mix(actual), donor, atol=1e-12, rtol=0)
        checks[arm] = {"conditional_payout_mix_max_abs_delta": float(np.abs(conditional_mix(actual) - donor).max()),
                       "mean_abs_win_delta_vs_W0": float(np.abs(p[arm] - p["W0"]).mean())}
    return checks, details


def slice_scores(split, details, model, validation, control, outputs):
    ids = split.query_ids
    targets = {t.query_id: t.probabilities for t in read_targets(validation / "targets.csv")}
    require_ids(targets, ids)
    truth = np.array([targets[q] for q in ids])
    reference = array_at(validation / "reference.csv", split)
    with (validation / "conditions.csv").open(newline="") as stream:
        condition_by_id = {r["query_id"]: r["condition"] for r in csv.DictReader(stream)}
    condition = np.array([condition_by_id[q] for q in ids])
    top_47 = details["top_pair"] == model.pairs.index(TARGET_PAIR)
    masks = {"top_pair_4_7": top_47, "other_top_pairs": ~top_47,
             "short_history": condition == "short_history",
             "short_history_top_pair_4_7": (condition == "short_history") & top_47}
    for cell, name in enumerate(CELL_NAMES):
        masks[f"top_cell_{name}"] = details["top_context"] == cell
        masks[f"top_pair_4_7_cell_{name}"] = top_47 & (details["top_context"] == cell)

    p = {name: array_at(path, split) for name, path in {"W0": control, **outputs}.items()}
    reference_error = np.square(reference - truth).sum(axis=1)
    model_errors = {name: np.square(pred - truth).sum(axis=1) for name, pred in p.items()}
    results = {}
    for label, mask in masks.items():
        report = {"queries": int(mask.sum()), "episodes": len(set(split.episode_ids[mask])), "models": {}}
        rng = np.random.default_rng(SEED)
        per_model_boot = {name: [] for name in p}
        if not np.any(mask):
            results[label] = report
            continue
        for name in p:
            report["models"][name] = {"p0_mse": float(np.square(p[name][mask, 0] - truth[mask, 0]).mean()),
                                       "condition_R": {}}
        for c in CONDITIONS:
            rows = mask & (condition == c)
            if not np.any(rows):
                continue
            _, inv, counts = np.unique(split.episode_ids[rows], return_inverse=True, return_counts=True)
            bootstrap = rng.integers(len(counts), size=(10000, len(counts)))
            denom = np.bincount(inv, weights=reference_error[rows]) / counts
            bootstrap_denom = denom[bootstrap].mean(axis=1).clip(1e-9)
            base = np.bincount(inv, weights=model_errors["W0"][rows]) / counts
            for name, err in model_errors.items():
                episode_err = np.bincount(inv, weights=err[rows]) / counts
                ratio = episode_err.mean() / max(denom.mean(), 1e-9)
                report["models"][name]["condition_R"][c] = float(ratio)
                per_model_boot[name].append((episode_err - base)[bootstrap].mean(axis=1) / bootstrap_denom)
        for name, score in report["models"].items():
            score["R"] = float(np.mean(list(score["condition_R"].values())))
            score["delta_R"] = score["R"] - report["models"]["W0"]["R"]
            score["delta_R_95"] = np.quantile(np.mean(per_model_boot[name], axis=0), [0.025, 0.975]).tolist()
        results[label] = report
    return results


def git_value(*args):
    result = subprocess.run(["git", *args], cwd=KIT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--test", action="store_true", help="Produce unscored test forecasts for W1a and W1b.")
    parser.add_argument("--forecast-only", action="store_true", help="Skip scoring exact released targets.")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    data = KIT / "public"
    started = perf_counter()
    with threadpool_limits(limits=2):
        train = labeled_input(data)
        strength, type_gate, selected, rows, selection = select_strength(train)
        write_rows(out / "train_grouped_predictions.csv", tuple(rows[0]), rows)
        write_json(out / "grouped_selection.json", selection)
        print(f"Selected local strength {strength}; train stage choice {selected}; W1c gate {type_gate}", flush=True)

        model = w0.read_checkpoint(FROZEN)
        fit = train_contrasts(model, train)
        historical_fit = json.loads((ORIGINAL.parent / "train_fit.json").read_text())
        np.testing.assert_allclose(fit.pooled[0], historical_fit["learned_contrast_11_minus_00"], atol=1e-12, rtol=0)
        write_json(out / "train_contrasts.json", {
            "pooled": fit.pooled.tolist(), "group_order": ["11_minus_00", "10_minus_01"],
            "direction_order": ["match_high", "differ_high"],
            "pairs": [list(pair) for pair in model.pairs], "pair_contrasts": fit.pair.tolist(),
            "type_order": list(TYPES), "effective_counts_by_type_pair_group_direction": fit.effective_counts.tolist(),
            **({"type_pair_contrasts": fit.typed().tolist()} if type_gate else {}),
        })
        variants = {"W1a": ("W1a", strength), "W1b": ("W1b", strength)}
        if type_gate:
            variants["W1c"] = ("W1c", strength)
        validation = input_only(data, "validation")
        check = causal_checks(model, validation, fit, variants)
        outputs = {name: out / f"{name}_validation_predictions.csv" for name in variants}
        parent = COMBINED.with_suffix(".csv")
        replay_start = perf_counter()
        check["validation_replay"], details = compose_forecasts(model, validation, fit, variants, parent, outputs)
        check["validation_preloaded_forecast_seconds"] = perf_counter() - replay_start
        cell_rows = []
        for i, query_id in enumerate(validation.query_ids):
            pair = model.pairs[details["top_pair"][i]]
            context = int(details["top_context"][i])
            cell_rows.append({"query_id": query_id, "episode_id": validation.episode_ids[i],
                              "top_pair": f"{pair[0]}-{pair[1]}",
                              "top_current_cell": CELL_NAMES[context] if context >= 0 else "missing",
                              **{f"top_no_flip_count_{cell}": int(value)
                                 for cell, value in zip(CELL_NAMES, details["top_no_flip_cell_counts"][i])},
                              **{f"posterior_weighted_current_count_{cell}": float(value)
                                 for cell, value in zip(CELL_NAMES, details["posterior_weighted_current_cell_count"][i])},
                              **{f"supported_posterior_mass_{cell}": float(value)
                                 for cell, value in zip(CELL_NAMES, details["supported_posterior_mass"][i])}})
        write_rows(out / "validation_cell_diagnostics.csv", tuple(cell_rows[0]), cell_rows)
        check["validation_centering_max_abs_delta"] = details["max_center_error"]
        input_paths = [data / f for f in ("metadata.json", "train/queries.csv", "train/history.csv", "train/labels.csv",
                                         "validation/queries.csv", "validation/history.csv")]
        test_paths = {}
        if args.test:
            input_paths.extend(data / "test" / f for f in ("queries.csv", "history.csv"))
            test = input_only(data, "test")
            test_paths = {name: out / f"{name}_test_predictions.csv" for name in variants}
            test_parent = COMBINED.with_name(COMBINED.name + "_test.csv")
            check["test_replay"], _ = compose_forecasts(model, test, fit, variants, test_parent, test_paths)
            input_paths.append(test_parent)
        check["elapsed_before_synthetic_seconds"] = perf_counter() - started
        write_json(out / "checks.json", check)
        synthetic = synthetic_checks(strength, type_gate)
        write_json(out / "synthetic_scores.json", synthetic)
        sources = [Path(__file__), ORIGINAL, EXPERIMENTS / "score.py", REPAIR / "run_repaired_comparison.py",
                   *(REPAIR / "models/common").glob("*.py"), REPAIR / "models/m3_lag_pair_flip/model.py",
                   *(KIT / "slot_forecasting").glob("*.py")]
        plan = {
            "round": json.loads((data / "metadata.json").read_text()),
            "branch": git_value("branch", "--show-current"), "commit": git_value("rev-parse", "--verify", "HEAD"),
            "seed": SEED, "class_order": [0, 1, 2, 5, 20], "thread_cap": 2,
            "versions": {"python": platform.python_version(),
                         **{pkg: version(pkg) for pkg in ("numpy", "scipy", "joblib", "scikit-learn", "threadpoolctl")}},
            "source_sha256": {str(path.relative_to(KIT)): sha256(path) for path in sources},
            "input_sha256": {str(path.relative_to(KIT)): sha256(path) for path in
                             [*input_paths, FOLDS, parent, FROZEN / "freeze.json", FROZEN / "models.joblib",
                              ORIGINAL.parent / "train_fit.json"]},
            "forecast_sha256": {path.name: sha256(path) for path in [*outputs.values(), *test_paths.values()]},
            "strength_grid": STRENGTHS, "selected_strength": strength, "pair_prior_pseudocount": PAIR_STRENGTH,
            "type_prior_pseudocount": TYPE_STRENGTH, "historical_cutoff_shifts": SHIFTS,
            "earlier_cutoff_signal": 0, "type_gate": type_gate, "train_selected_win_head": selected,
            "frozen_m3_kappa": model.params["kappa"],
            "elapsed_forecast_and_checks_seconds": perf_counter() - started,
        }
        write_json(out / "plan.json", plan)
        if not args.forecast_only:
            # Exact probabilities are used only after all candidate files and choices are frozen.
            scores = compare(data / "validation", parent, outputs, 60)
            scores["slice_R_definition"] = "Equal-weight available condition ratios using episode means within the slice for both forecast and reference."
            scores["slices"] = slice_scores(validation, details, model, data / "validation", parent, outputs)
            scores["pair_and_type_increment_vs_W1a"] = compare(
                data / "validation", outputs["W1a"], {name: path for name, path in outputs.items() if name != "W1a"}, 60
            )["models"]
            if not np.isclose(scores["models"]["control"]["points"], 56.341022, rtol=0, atol=5e-7):
                raise ValueError("Official W0 score differed from the published hybrid.")
            write_json(out / "scores.json", scores)
            for name, result in scores["models"].items():
                print(f"{name}: {result['points']:.6f}, R {result['R']:.8f}", flush=True)


if __name__ == "__main__":
    main()
