"""Score only forecast files whose hashes were frozen in the prior phase."""
import csv
import json

import numpy as np

from replay import DATA, OUT, ROOT, load_w2, trace


def slices(w1, split, condition, forecasts, baselines):
    with (OUT / "frozen_validation_slices.csv").open(newline="") as stream:
        definitions = {row["query_id"]: row for row in csv.DictReader(stream)}
    pair = np.asarray([definitions[q]["top_pair"] for q in split.query_ids])
    pair_mass = np.asarray([float(definitions[q]["top_pair_mass"]) for q in split.query_ids])
    cell = np.asarray([int(definitions[q]["top_cell"]) for q in split.query_ids])
    target = pair == "4-7"
    short = condition == "short_history"
    masks = {
        "overall": np.ones(split.n, bool), "parent_top_pair_4_7": target, "other_top_pairs": ~target,
        "parent_top_pair_mass_ge_0.90": pair_mass >= .90,
        "parent_top_pair_mass_ge_0.99": pair_mass >= .99,
        "parent_top_pair_4_7_mass_ge_0.99": target & (pair_mass >= .99),
        "short_history": short, "short_history_parent_top_pair_4_7": short & target,
    }
    for i, name in enumerate(w1.CELL_NAMES):
        masks[f"parent_raw_cell_{name}"] = cell == i
        masks[f"parent_top_pair_4_7_cell_{name}"] = target & (cell == i)
    truth_by_id = {t.query_id: t.probabilities for t in w1.read_targets(DATA / "validation/targets.csv")}
    truth = np.asarray([truth_by_id[q] for q in split.query_ids])
    reference = w1.array_at(DATA / "validation/reference.csv", split)
    errors_by_class = {name: np.square(w1.array_at(path, split) - truth) for name, path in forecasts.items()}
    reference_error = np.square(reference - truth).sum(axis=1)
    outputs = []
    for slice_name, mask in masks.items():
        if not mask.any():
            continue
        rng = np.random.default_rng(w1.SEED)
        arm_ratios = {name: [] for name in forecasts}
        arm_bootstrap = {name: [] for name in forecasts}
        for stratum in w1.CONDITIONS:
            rows = mask & (condition == stratum)
            if not rows.any():
                continue
            _, inv, n = np.unique(split.episode_ids[rows], return_inverse=True, return_counts=True)
            sample = rng.integers(len(n), size=(10_000, len(n)))
            reference_eps = np.bincount(inv, weights=reference_error[rows]) / n
            den = max(reference_eps.mean(), 1e-9)
            boot_den = reference_eps[sample].mean(axis=1).clip(1e-9)
            for name, squared in errors_by_class.items():
                episode_sse = np.bincount(inv, weights=squared[rows].sum(axis=1)) / n
                arm_ratios[name].append(episode_sse.mean() / den)
                arm_bootstrap[name].append(episode_sse[sample].mean(axis=1) / boot_den)
        for name, squared in errors_by_class.items():
            contributions = squared[mask].mean(axis=0)
            r = float(np.mean(arm_ratios[name]))
            comparison = baselines.get(name, "hybrid")
            row = {"slice": slice_name, "queries": int(mask.sum()), "episodes": len(set(split.episode_ids[mask])),
                   "arm": name, "R": r, "matched_control": comparison}
            for label, base in (("hybrid", "hybrid"), ("matched", comparison)):
                ci = np.quantile(np.mean(arm_bootstrap[name], axis=0) - np.mean(arm_bootstrap[base], axis=0),
                                 [.025, .975])
                row.update({f"delta_R_vs_{label}": r - np.mean(arm_ratios[base]),
                            f"ci_low_vs_{label}": ci[0], f"ci_high_vs_{label}": ci[1]})
            row.update(dict(zip(("p0_mse", "p1_mse", "p2_mse", "p5_mse", "p20_mse"), contributions)))
            outputs.append(row)
    w1.write_rows(OUT / "slices.csv", tuple(outputs[0]), outputs)


def main():
    trace("score")
    w2 = load_w2()
    w1 = w2.w1
    freeze = json.loads((OUT / "forecast_freeze.json").read_text())
    assert w1.sha256(OUT / "frozen_fit.json") == freeze["fit_sha256"]
    for filename, digest in freeze["forecast_sha256"].items():
        assert w1.sha256(OUT / filename) == digest
    validation = DATA / "validation"
    split = w1.input_only(DATA, "validation")
    with (validation / "conditions.csv").open(newline="") as stream:
        by_query = {row["query_id"]: row["condition"] for row in csv.DictReader(stream)}
    condition = np.asarray([by_query[q] for q in split.query_ids])
    arms = ("W1a", "W1b", "W2", "E3Frozen", "E3Full", "E3Shared")
    forecasts = {name: OUT / f"validation_{name}.csv" for name in arms}
    hybrid = w1.COMBINED.with_suffix(".csv")
    m3 = w1.FROZEN / "m3/validation_predictions.csv"
    matched = {"W1b": "W1a", "W2": "W1a", "E3Frozen": "W1a", "E3Full": "E3Frozen", "E3Shared": "E3Full"}
    for name, baseline in (("hybrid", hybrid), ("m3", m3)):
        w1.write_json(OUT / f"scores_vs_{name}.json", w1.compare(validation, baseline, forecasts, 60))
    matched_results = {base: w1.compare(validation, forecasts[base], {name: forecasts[name] for name, control
                       in matched.items() if control == base}, 60) for base in dict.fromkeys(matched.values())}
    matched_results["W1a_operating_control"] = w1.compare(validation, forecasts["W1a"], {
        name: forecasts[name] for name in ("E3Full", "E3Shared")}, 60)
    w1.write_json(OUT / "scores_vs_matched.json", matched_results)
    slices(w1, split, condition, {"hybrid": hybrid, "M3": m3, **forecasts}, matched)


if __name__ == "__main__":
    main()
