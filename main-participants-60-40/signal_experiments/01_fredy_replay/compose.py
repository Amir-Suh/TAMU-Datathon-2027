"""Compose fixed, equally weighted forecast variants and rescore the main split."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
KIT = HERE.parents[1]
sys.path[:0] = [str(HERE.parent), str(KIT)]

from score import BASELINE, compare
from slot_forecasting.io import read_queries, read_submission, write_submission

REPAIR = KIT / "m5_m6_repair/results/repaired-m5-m6"


def average(*forecasts):
    return np.mean(forecasts, axis=0)


def conditional_mix(probabilities):
    positive = probabilities[:, 1:]
    totals = positive.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("Conditional payout mix requires positive win mass.")
    return positive / totals


def with_mix(win_source, positive_mix):
    return np.column_stack([win_source[:, 0], (1 - win_source[:, :1]) * positive_mix])


def replace_conditional_mix(win_source, payout_source):
    return with_mix(win_source, conditional_mix(payout_source))


def compose(directory, ids, m3, m6, m6_previous, *, partial):
    def read(path):
        mapping = read_submission(path, ids)
        return np.array([mapping[key] for key in ids])

    def write(name, probabilities):
        path = directory / f"{name}.csv"
        write_submission(path, dict(zip(ids, probabilities)))
        paths[name] = path

    directory.mkdir(parents=True, exist_ok=True)
    pair_names = ("pair_trans_64_l15", "pair_l31")
    seed_names = tuple(f"gru128_nomsg_seed_{i}" for i in range(1, 8))
    auxiliary = ("gru128_factor_seed_1", "gru128_factor_seed_2", "gru128_factor_seed_3")
    paths = {
        name: directory / f"{name}.csv"
        for name in (*pair_names, *seed_names, *auxiliary)
        if (directory / f"{name}.csv").exists()
    }
    if not partial:
        missing = set((*pair_names, *seed_names)) - paths.keys()
        if missing:
            raise FileNotFoundError(f"Missing replay forecasts: {sorted(missing)}")

    seeds = [read(paths[name]) for name in seed_names if name in paths]
    suffix = "" if len(seeds) == len(seed_names) else f"_partial_{len(seeds)}"
    seed_mean = average(*seeds) if seeds else None
    if seeds:
        write(f"gru128_nomsg_mean{suffix}", seed_mean)
    factors = [read(paths[name]) for name in auxiliary if name in paths]
    if factors:
        write(f"gru128_factor_mean_{len(factors)}", average(*factors))

    m3_p = read(m3)
    for pair_name in pair_names:
        if pair_name not in paths:
            continue
        pair = read(paths[pair_name])
        write(f"m3_win_{pair_name}_size", replace_conditional_mix(m3_p, pair))
        if m6 is not None:
            write(f"{pair_name}_repaired_m6", average(pair, read(m6)))
        if m6_previous is not None:
            write(f"{pair_name}_previous_m6", average(pair, read(m6_previous)))
        if seed_mean is None:
            continue
        fredy = average(pair, seed_mean)
        write(f"{pair_name}_gru{suffix}", fredy)
        write(f"m3_win_{pair_name}_gru_size{suffix}", replace_conditional_mix(m3_p, fredy))
        write(
            f"m3_win_{pair_name}_gru_equal_size{suffix}",
            with_mix(m3_p, average(conditional_mix(pair), conditional_mix(seed_mean))),
        )
        for m6_name, m6_path in (("repaired", m6), ("previous", m6_previous)):
            if m6_path is not None:
                blend = average(pair, seed_mean, read(m6_path))
                write(f"{pair_name}_gru_m6_{m6_name}{suffix}", blend)
                write(
                    f"m3_win_{pair_name}_gru_m6_{m6_name}_size{suffix}",
                    replace_conditional_mix(m3_p, blend),
                )
    if len(factors) == len(auxiliary) and "pair_l31" in paths:
        pair = read(paths["pair_l31"])
        factored_mean = average(*factors)
        donor = average(pair, factored_mean)
        write("pair_l31_factor_gru", donor)
        write("m3_win_pair_l31_factor_gru_size", replace_conditional_mix(m3_p, donor))
        if m6 is not None:
            write("pair_l31_factor_gru_m6_repaired", average(pair, factored_mean, read(m6)))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    test = KIT / "public/test"
    validation = KIT / "public/validation"
    def ids(directory):
        return [q.query_id for q in read_queries(directory / "queries.csv")]
    paths = compose(
        HERE / "forecasts",
        ids(validation),
        BASELINE,
        REPAIR / "m6/validation_predictions.csv",
        REPAIR / "previous/m6/validation_predictions.csv",
        partial=args.allow_partial,
    )
    compose(
        HERE / "test_forecasts",
        ids(test),
        REPAIR / "m3/submission.csv",
        REPAIR / "m6/submission.csv",
        None,
        partial=args.allow_partial,
    )
    paths["m6_repaired"] = REPAIR / "m6/validation_predictions.csv"
    paths["m6_previous"] = REPAIR / "previous/m6/validation_predictions.csv"
    results = compare(validation, BASELINE, paths, 60)
    models = results["models"]
    control = models["control"]
    for model in models.values():
        model["path"] = str(Path(model["path"]).relative_to(KIT))
        model["delta_points"] = model["points"] - control["points"]
        model["condition_delta_R"] = {
            condition: ratio - control["condition_R"][condition]
            for condition, ratio in model["condition_R"].items()
        }
    results["head_swap_vs_full_blend"] = {}
    for pair, m6 in (("pair_l31", "repaired"), ("pair_trans_64_l15", "previous")):
        full = f"{pair}_gru_m6_{m6}"
        same_mix = f"m3_win_{pair}_gru_m6_{m6}_size"
        fredy_mix = f"m3_win_{pair}_gru_size"
        if full not in paths:
            continue
        direct = compare(validation, paths[full], {key: paths[key] for key in (same_mix, fredy_mix)}, 60)
        results["head_swap_vs_full_blend"][full] = {
            key: {
                "delta_R": row["delta_R"],
                "delta_R_95": row["delta_R_95"],
                "delta_points": row["points"] - models[full]["points"],
            }
            for key, row in direct["models"].items()
            if key != "control"
        }
    (HERE / "scores.json").write_text(json.dumps(results, indent=2) + "\n")
    for name, model in results["models"].items():
        print(f"{name:65} {model['points']:.6f}  {model['delta_R']:+.6f}")


if __name__ == "__main__":
    main()
