"""Update M3 evidence with a normalized ordered-cell allocation tilt."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import gammaln, logit, logsumexp
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
NEXT = HERE.parent
sys.path.insert(0, str(NEXT))
from synthetic import generate, mask_history

W1 = NEXT / "W1"
spec = importlib.util.spec_from_file_location("w1_ordered_cells", W1 / "run.py")
w1 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = w1
spec.loader.exec_module(w1)
STAGE = "W1a"
CHUNK = 2048


def log_choose_table(max_n):
    n = np.arange(max_n + 1)[:, None]
    k = np.arange(max_n + 1)[None, :]
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def log_allocation_ratio(n_left, n_right, wins_left, wins_right, log_odds, choose):
    """Exact log tilted-to-central allocation ratio for each fixed count design."""
    ratio = np.zeros(len(n_left))
    for start in range(0, len(ratio), CHUNK):
        ix = slice(start, start + CHUNK)
        nl, nr, kl, kr, odds = (values[ix] for values in (n_left, n_right, wins_left, wins_right, log_odds))
        total = kl + kr
        lower, upper = np.maximum(0, total - nl), np.minimum(nr, total)
        j = lower[:, None] + np.arange(1 + (upper - lower).max())
        valid = j <= upper[:, None]
        j = np.minimum(j, upper[:, None])
        log_weight = choose[nl[:, None], total[:, None] - j] + choose[nr[:, None], j]
        log_weight += odds[:, None] * (j - kr[:, None])
        log_weight[~valid] = -np.inf
        value = choose[nl + nr, total] - logsumexp(log_weight, axis=1)
        ratio[ix] = np.where((lower == upper) | (odds == 0), 0.0, value)
    return ratio


def odds_from_pooled(model, fit):
    """Convert the two training contrasts to symmetric child logits around each M3 context rate."""
    match_rates = np.column_stack((model.hi, model.lo))
    differ_rates = np.column_stack((model.lo, model.hi))
    p = np.stack((match_rates, differ_rates), axis=1)
    left, right = w1.centered_children(p, 0.5, fit.pooled[None, :, :])
    return logit(np.clip(right, 1e-6, 1 - 1e-6)) - logit(np.clip(left, 1e-6, 1 - 1e-6))


def update_evidence(model, split, base, odds):
    w = w1.w0.win_history(split)
    core = base["core"]
    llp = core["llp"].copy()
    possible_tau = np.isfinite(core["log_tau_prior"])
    choose = log_choose_table(w.shape[1])
    max_abs = 0.0
    for i, pair in enumerate(model.pairs):
        n, k, _ = w1.ordered_counts(w, split.cutoff, pair, model.tau_grid)
        for group, (left, right) in enumerate(w1.CELLS):
            n_left, n_right = n[..., left], n[..., right]
            k_left, k_right = k[..., left], k[..., right]
            total = k_left + k_right
            usable = (possible_tau & (n_left > 0) & (n_right > 0) & (total > 0) &
                      (total < n_left + n_right))
            rows, tau = np.where(usable)
            if not len(rows):
                continue
            for direction in (0, 1):
                log_odds = odds[split.context[rows], group, direction]
                ratio = log_allocation_ratio(n_left[usable], n_right[usable], k_left[usable],
                                             k_right[usable], log_odds, choose)
                llp[rows, i, direction, tau] += ratio
                max_abs = max(max_abs, float(np.abs(ratio).max()))
    updated_core = dict(core, llp=llp)
    updated_independent = model._independent(updated_core)
    return dict(base, core=updated_core, ind=updated_independent, p_win=updated_independent["p_win"]), max_abs


class PosteriorReplay:
    """Make W1 apply its unchanged forecasts to a supplied hypothesis posterior."""
    def __init__(self, model, result):
        self.pairs, self.tau_grid = model.pairs, model.tau_grid
        self.result = result

    def _infer(self, split):
        return self.result


def pair_entropy(result):
    pair_post = result["ind"]["post"].sum(axis=(2, 3))
    mass = np.column_stack((pair_post, result["ind"]["post0"]))
    entropy = -np.sum(mass * np.log(np.clip(mass, 1e-300, 1)), axis=1)
    return pair_post, entropy


def paired_forecasts(model, split, fit, strength):
    variant = {STAGE: (STAGE, strength)}
    original = model._infer(split)
    updated, max_tilt = update_evidence(model, split, original, odds_from_pooled(model, fit))
    p1, details = w1.forecast(PosteriorReplay(model, original), split, fit, variant)
    p2, changed = w1.forecast(PosteriorReplay(model, updated), split, fit, variant)
    post1, entropy1 = pair_entropy(original)
    post2, entropy2 = pair_entropy(updated)
    posterior1 = original["ind"]["post"]
    posterior2 = updated["ind"]["post"]
    return p1[STAGE], p2[STAGE], {
        "details": details, "updated_details": changed, "max_abs_single_group_log_tilt": max_tilt,
        "pair_entropy_w1": entropy1, "pair_entropy_w2": entropy2,
        "top_pair_mass_w1": post1.max(axis=1), "top_pair_mass_w2": post2.max(axis=1),
        "null_mass_w1": original["ind"]["post0"], "null_mass_w2": updated["ind"]["post0"],
        "flip_mass_w1": posterior1[..., 1:].sum(axis=(1, 2, 3)),
        "flip_mass_w2": posterior2[..., 1:].sum(axis=(1, 2, 3)),
        "map_flip_tick_w1": np.r_[0, model.tau_grid][posterior1.sum(axis=(1, 2)).argmax(axis=1)],
        "map_flip_tick_w2": np.r_[0, model.tau_grid][posterior2.sum(axis=(1, 2)).argmax(axis=1)],
        "map_direction_w1": posterior1.sum(axis=(1, 3)).argmax(axis=1),
        "map_direction_w2": posterior2.sum(axis=(1, 3)).argmax(axis=1),
    }


def check_exact_normalization():
    choose = log_choose_table(12)
    max_error = 0.0
    checks = 0
    for n_left, n_right in ((0, 3), (1, 2), (3, 3), (3, 4)):
        n = n_left + n_right
        for k_total in range(n + 1):
            winners = [bits for bits in itertools.product((0, 1), repeat=n) if sum(bits) == k_total]
            k_right = np.array([sum(bits[n_left:]) for bits in winners])
            for odds in (-1.5, 0.0, 0.7):
                ratio = log_allocation_ratio(np.full(len(winners), n_left), np.full(len(winners), n_right),
                                             k_total - k_right, k_right, np.full(len(winners), odds), choose)
                actual = np.exp(ratio) / len(winners)
                expected = np.exp(odds * k_right)
                expected /= expected.sum()
                max_error = max(max_error, float(np.abs(expected - actual).max()), abs(1 - actual.sum()))
                np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
                if not odds:
                    np.testing.assert_array_equal(ratio, 0)
                checks += 1
    return {"fixed_design_explicit_outcome_distributions": checks, "max_normalization_or_oracle_error": max_error}


def causal_checks(model, split, fit, strength):
    rows = np.arange(16)
    sample = w1.subset_split(split, rows)
    original = model._infer(sample)
    updated, tilt = update_evidence(model, sample, original, np.zeros_like(odds_from_pooled(model, fit)))
    assert tilt == 0
    for key in ("post", "post0", "p_win"):
        np.testing.assert_array_equal(updated["ind"][key], original["ind"][key])
    w1a, w2, _ = paired_forecasts(model, sample, fit, strength)
    padded = replace(sample, X=np.pad(sample.X, ((0, 0), (0, 13)), constant_values=w1.MISSING))
    w1_padded, w2_padded, _ = paired_forecasts(model, padded, fit, strength)
    reversed_rows = rows[::-1]
    w1_reversed, w2_reversed, _ = paired_forecasts(model, w1.subset_split(sample, reversed_rows), fit, strength)
    for base, padding, ordering in ((w1a, w1_padded, w1_reversed), (w2, w2_padded, w2_reversed)):
        np.testing.assert_array_equal(base, padding)
        np.testing.assert_array_equal(base, ordering[reversed_rows])
    old_cutoff = sample.cutoff - 16
    later = np.arange(sample.X.shape[1])[None, :] >= old_cutoff[:, None]
    altered = sample.X.copy()
    altered[later] = 4
    earlier = w1.prefix(replace(sample, labels=np.zeros(sample.n, dtype=int)), 16)
    perturbed = w1.prefix(replace(sample, X=altered, labels=np.ones(sample.n, dtype=int)), 16)
    _, early, _ = paired_forecasts(model, earlier, fit, strength)
    _, early_perturbed, _ = paired_forecasts(model, perturbed, fit, strength)
    np.testing.assert_array_equal(early, early_perturbed)
    return {"zero_contrast_exact_posterior_replay": True, "exact_padding_and_row_order_invariance": True,
            "earlier_forecast_ignores_at_cutoff_and_future": True, **check_exact_normalization()}


def synthetic_scores(strength):
    results = {}
    for four_cells in (False, True):
        stable_train = generate("stable", episodes=32, seed=38102, four_cells=four_cells)
        flip_train = generate("flip", episodes=32, seed=38103, lead=20, four_cells=four_cells)
        a, b = stable_train.split, flip_train.split
        fields = {
            "query_ids": a.query_ids + b.query_ids, "machine_ids": a.machine_ids + b.machine_ids,
            **{name: np.concatenate((getattr(a, name), getattr(b, name))) for name in
               ("episode_ids", "machine_type", "signal", "cutoff", "X", "labels", "conditions", "targets")},
        }
        train = replace(a, **fields)
        model = w1.refit_m3(train, tune=False)
        contrasts = w1.train_contrasts(model, train)
        law_results = {"pooled_contrasts": contrasts.pooled.tolist(), "cases": {}}
        for scenario in ("stable", "flip"):
            for lead in ((16,) if scenario == "stable" else (8, 16, 40)):
                fixture = generate(scenario, episodes=24, seed=38104, lead=lead, four_cells=four_cells)
                for older in (0, 24, None):
                    split = fixture.split if older is None else mask_history(fixture.split, older)
                    a, b, details = paired_forecasts(model, split, contrasts, strength)
                    exact = split.targets
                    size = w1.conditional_mix(exact)
                    per_arm = {}
                    for suffix, name, p, pair in (
                        ("w1", "W1a", a, details["details"]["top_pair"]),
                        ("w2", "W2", b, details["updated_details"]["top_pair"]),
                    ):
                        prediction = w1.combine(p, size)
                        map_tick = details[f"map_flip_tick_{suffix}"]
                        per_arm[name] = {
                            "p0_mse": float(np.mean(np.square(prediction[:, 0] - exact[:, 0]))),
                            "five_class_sse": float(np.mean(np.square(prediction - exact).sum(axis=1))),
                            "true_pair_recovery": float(np.all(np.array(model.pairs)[pair] == fixture.pairs, axis=1).mean()),
                            "true_direction_recovery": float(np.mean(details[f"map_direction_{suffix}"] == fixture.polarity)),
                            "pair_entropy_nats": float(details[f"pair_entropy_{suffix}"].mean()),
                            "mean_flip_mass": float(details[f"flip_mass_{suffix}"].mean()),
                            "map_flip_selected_share": float(np.mean(map_tick > 0)),
                        }
                        if scenario == "flip":
                            per_arm[name]["map_flip_within_4_ticks_share"] = float(np.mean(
                                (map_tick > 0) & (np.abs(map_tick - fixture.change) <= 4)))
                    law_results["cases"][f"{scenario}_{lead}_older_{older if older is not None else 'all'}"] = per_arm
        results["ordered" if four_cells else "parity"] = law_results
    return {"fit_episodes_per_law": 64, "held_episodes_per_case": 24, "fit_seeds": [38102, 38103],
            "held_seed": 38104, "fit_kappa": 8, "history_recent": 40,
            "size_scope": "The known-law exact conditional mix is shared by both arms to isolate the win head.",
            "laws": results}


def grouped_selection(train, strength):
    with w1.FOLDS.open(newline="") as stream:
        by_episode = {r["episode_id"]: int(r["fold"]) for r in csv.DictReader(stream) if r["split"] == "train"}
    w1.require_ids(by_episode, set(train.episode_ids))
    fold = np.array([by_episode[e] for e in train.episode_ids])
    rows, fit_info = [], []
    for held_fold in sorted(set(fold)):
        fitting = w1.subset_split(train, np.flatnonzero(fold != held_fold))
        held = w1.subset_split(train, np.flatnonzero(fold == held_fold))
        fitting_ids, held_ids = set(fitting.episode_ids), set(held.episode_ids)
        assert fitting_ids.isdisjoint(held_ids)
        print(f"W2 refitting M3 for train fold {held_fold}", flush=True)
        model = w1.refit_m3(fitting, tune=True)
        fit = w1.train_contrasts(model, fitting)
        fit_info.append({"fold": int(held_fold), "fit_episodes": len(fitting_ids), "held_episodes": len(held_ids),
                         "fit_episode_sha256": w1.id_hash(sorted(fitting_ids)),
                         "held_episode_sha256": w1.id_hash(sorted(held_ids)), "m3_kappa": model.params["kappa"]})
        for shift in w1.SHIFTS:
            frame = w1.prefix(held, shift)
            a, b, details = paired_forecasts(model, frame, fit, strength)
            pair = details["details"]["top_pair"] == model.pairs.index(w1.TARGET_PAIR)
            for i in np.flatnonzero(frame.labels != w1.MISSING):
                rows.append({"query_id": frame.query_ids[i], "episode_id": frame.episode_ids[i],
                             "fold": int(held_fold), "shift": shift, "win": int(frame.labels[i] > 0),
                             "W1a_top_pair_4_7": bool(pair[i]), "W1a": float(a[i]), "W2": float(b[i])})
    y = np.array([r["win"] for r in rows], dtype=bool)
    episodes = np.array([r["episode_id"] for r in rows])
    endpoint = np.array([r["shift"] == 0 for r in rows])
    target_pair = np.array([r["W1a_top_pair_4_7"] for r in rows])
    metrics = {}
    for arm in ("W1a", "W2"):
        p = np.array([r[arm] for r in rows])
        metrics[arm] = {k: w1.grouped_metrics(episodes, y, p, mask) for k, mask in {
            "all": np.ones(len(y), dtype=bool), "top_pair_4_7": target_pair, "actual_cutoff": endpoint,
            "earlier_cutoffs": ~endpoint}.items()}
    frozen_rows = {}
    with (W1 / "train_grouped_predictions.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            frozen_rows[row["query_id"], int(row["shift"])] = float(row[f"W1a_{strength}"])
    replay_delta = max(abs(row["W1a"] - frozen_rows[row["query_id"], row["shift"]]) for row in rows)
    if replay_delta > 1e-12 or len(frozen_rows) != len(rows):
        raise ValueError("W2 grouped W1a control did not replay.")
    return rows, {"refit_scope": "All upstream M3 win and W1 pooled parameters refit inside each fold.",
                  "folds": fit_info, "metrics": metrics, "W1a_max_abs_replay_delta": replay_delta}


def validation_diagnostics(split, details, model, data):
    with (data / "validation/conditions.csv").open(newline="") as stream:
        by_id = {r["query_id"]: r["condition"] for r in csv.DictReader(stream)}
    conditions = np.array([by_id[q] for q in split.query_ids])
    w1_top = details["details"]["top_pair"]
    w2_top = details["updated_details"]["top_pair"]
    masks = {"all": np.ones(split.n, dtype=bool), "short_history": conditions == "short_history",
             "W1a_top_pair_4_7": w1_top == model.pairs.index(w1.TARGET_PAIR)}
    output = {}
    for name, mask in masks.items():
        output[name] = {"queries": int(mask.sum()), "top_pair_changes": int((w1_top[mask] != w2_top[mask]).sum()),
                        **{k: float(v[mask].mean()) for k, v in details.items() if isinstance(v, np.ndarray)}}
    return output


def write_forecast(model, split, fit, strength, frozen, control, out):
    a, b, details = paired_forecasts(model, split, fit, strength)
    size = w1.conditional_mix(w1.array_at(frozen, split))
    frozen_p = w1.array_at(control, split)
    base = w1.combine(a, size)
    candidate = w1.combine(b, size)
    np.testing.assert_allclose(base, frozen_p, atol=1e-12, rtol=0)
    w1.write_submission(out, dict(zip(split.query_ids, candidate)))
    readback = w1.array_at(out, split)
    donor_delta = float(np.abs(w1.conditional_mix(readback) - size).max())
    if donor_delta > 1e-12:
        raise ValueError("W2 conditional payout mix changed.")
    return {"W1a_max_abs_replay_delta": float(np.abs(frozen_p - base).max()),
            "conditional_payout_mix_max_abs_delta": donor_delta,
            "mean_abs_win_change_vs_W1a": float(np.abs(a - b).mean()),
            "max_abs_single_group_log_tilt": details["max_abs_single_group_log_tilt"]}, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--forecast-only", action="store_true")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    data = w1.KIT / "public"
    started = perf_counter()
    with threadpool_limits(limits=2):
        w1_manifest = json.loads((W1 / "plan.json").read_text())
        if w1_manifest["train_selected_win_head"] != STAGE:
            raise ValueError("W2 requires a frozen W1a training-stage winner.")
        strength = w1_manifest["selected_strength"]
        model = w1.w0.read_checkpoint(w1.FROZEN)
        train = w1.labeled_input(data)
        fit = w1.train_contrasts(model, train)
        expected_fit = json.loads((W1 / "train_contrasts.json").read_text())["pooled"]
        np.testing.assert_allclose(fit.pooled, expected_fit, rtol=0, atol=1e-12)
        validation = w1.input_only(data, "validation")
        smoke = w1.subset_split(validation, np.flatnonzero(np.isin(validation.episode_ids,
                                                                  np.unique(validation.episode_ids)[:12])))
        timer = perf_counter()
        paired_forecasts(model, smoke, fit, strength)
        smoke_seconds = perf_counter() - timer
        print(f"W2 12-episode smoke in {smoke_seconds:.2f}s", flush=True)
        checks = causal_checks(model, validation, fit, strength)
        rows, grouped = grouped_selection(train, strength)
        w1.write_rows(out / "train_grouped_predictions.csv", tuple(rows[0]), rows)
        w1.write_json(out / "grouped_metrics.json", grouped)
        baseline = w1.COMBINED.with_suffix(".csv")
        w1_csv = W1 / "W1a_validation_predictions.csv"
        w2_csv = out / "validation_predictions.csv"
        timing = perf_counter()
        checks["validation"], details = write_forecast(model, validation, fit, strength, baseline, w1_csv, w2_csv)
        checks["validation_preloaded_forecast_seconds"] = perf_counter() - timing
        checks["smoke_12_episode_seconds"] = smoke_seconds
        diagnostics = validation_diagnostics(validation, details, model, data)
        w1.write_json(out / "posterior_diagnostics.json", diagnostics)
        input_paths = [W1 / "plan.json", W1 / "train_contrasts.json", W1 / "train_grouped_predictions.csv", w1_csv,
                       baseline, w1.FOLDS, w1.FROZEN / "models.joblib", w1.FROZEN / "freeze.json"]
        output_paths = [w2_csv]
        if args.test:
            test = w1.input_only(data, "test")
            test_baseline = w1.COMBINED.with_name(w1.COMBINED.name + "_test.csv")
            test_w1 = W1 / "W1a_test_predictions.csv"
            test_w2 = out / "test_predictions.csv"
            checks["test"], _ = write_forecast(model, test, fit, strength, test_baseline, test_w1, test_w2)
            input_paths.extend([test_baseline, test_w1, data / "test/queries.csv", data / "test/history.csv"])
            output_paths.append(test_w2)
        w1.write_json(out / "checks.json", checks)
        w1.write_json(out / "synthetic_scores.json", synthetic_scores(strength))
        w1.write_json(out / "plan.json", {
            "round": w1_manifest["round"], "branch": w1_manifest["branch"], "commit": w1_manifest["commit"],
            "threads": 2, "class_order": [0, 1, 2, 5, 20], "versions": w1_manifest["versions"],
            "w1a_selected_strength": strength, "w1a_pooled_contrasts": fit.pooled.tolist(),
            "context_log_odds_right_minus_left": odds_from_pooled(model, fit).tolist(),
            "allocation_scope": "Exact fixed-exposure conditional Bayes-factor tilt. Lag-cell counts are treated as fixed for the allocation, not as a chronological generative law.",
            "w1_upstream_source_sha256": w1_manifest["source_sha256"],
            "source_sha256": {str(p.relative_to(w1.KIT)): w1.sha256(p) for p in [Path(__file__), W1 / "run.py", NEXT / "synthetic.py"]},
            "data_input_sha256": {str(p.relative_to(w1.KIT)): w1.sha256(p) for p in [
                data / "metadata.json", data / "train/queries.csv", data / "train/history.csv",
                data / "train/labels.csv", data / "validation/queries.csv", data / "validation/history.csv"]},
            "other_input_sha256": {str(p.relative_to(w1.KIT)): w1.sha256(p) for p in input_paths},
            "forecast_sha256": {p.name: w1.sha256(p) for p in output_paths},
            "elapsed_before_scoring_seconds": perf_counter() - started,
        })
        if not args.forecast_only:
            scores = w1.compare(data / "validation", w1_csv, {"W2": w2_csv}, 60)
            scores["comparison_vs_frozen_hybrid"] = w1.compare(
                data / "validation", baseline, {"W1a": w1_csv, "W2": w2_csv}, 60)["models"]
            slices = w1.slice_scores(validation, details["details"], model, data / "validation", w1_csv, {"W2": w2_csv})
            for item in slices.values():
                if "W0" in item["models"]:
                    item["models"]["W1a"] = item["models"].pop("W0")
            scores["slices_vs_W1a"] = slices
            scores["slice_R_definition"] = "Equal-weight available condition ratios using forecast and reference episode means within the slice."
            w1.write_json(out / "scores.json", scores)
            for arm, score in scores["models"].items():
                print(f"{arm}: {score['points']:.6f}, R {score['R']:.8f}", flush=True)


if __name__ == "__main__":
    main()
