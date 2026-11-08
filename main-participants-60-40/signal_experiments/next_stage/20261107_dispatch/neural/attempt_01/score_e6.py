"""Score only frozen E6 predictions and report matched, slice, and signal contrasts."""
import argparse
import json
from pathlib import Path
from time import perf_counter

from run import (ARMS, EXP, FREDY, KIT, PUBLIC, SEEDS, digest, eligible_prefixes, input_only, np,
                 read_rows, read_submission, write_json)
from signal_experiments.score import BASELINE as M3, compare
from slot_forecasting.io import read_targets, write_rows

HYBRID = EXP / "combined/m3_match_split_win_pair_l31_factor_gru_size.csv"
ARCHIVED_GRU = FREDY / "forecasts/gru128_factor_mean_3.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    args = parser.parse_args()
    start = perf_counter()
    frozen = args.frozen
    freeze = json.loads((frozen / "forecast_freeze.json").read_text())
    for name, expected in freeze.items():
        assert digest(frozen / name) == expected, name
    out = frozen.parent
    data = PUBLIC / "validation"
    split = input_only(PUBLIC, "validation")
    candidates = {name[len("validation_"):-4]: frozen / name
                  for name in freeze if name.startswith("validation_") and name.endswith(".csv")}
    plain = {n: p for n, p in candidates.items() if not n.endswith("_mask24_40")}
    masked = {n: p for n, p in candidates.items() if n.endswith("_mask24_40")}
    external = {"archived_factored_gru": ARCHIVED_GRU}
    comparisons = {
        "scores_vs_hybrid": (HYBRID, {**external, **plain}),
        "scores_vs_m3": (M3, {"archived_hybrid": HYBRID, **external, **plain}),
        "scores_vs_matched": (plain["factored"], {n: p for n, p in plain.items() if n != "factored"}),
        "scores_vs_structural": (plain["structural"], {n: p for n, p in plain.items() if n != "structural"}),
        "scores_mask_vs_matched": (masked["signal_offset_mask24_40"], masked),
        "scores_aug_vs_unaug": (plain["signal_offset"], {"signal_offset_mask": plain["signal_offset_mask"]}),
        "scores_mask_vs_unmasked": (plain["signal_offset"], {**masked, "signal_offset_mask": plain["signal_offset_mask"]}),
    }
    for name, (baseline, forecasts) in comparisons.items():
        write_json(out / f"{name}.json", compare(data, baseline, forecasts, 60))

    matched_seeds = {}
    for seed in SEEDS:
        key = f"_seed{seed}"
        first = compare(data, plain["factored" + key], {a: plain[a + key] for a in ARMS[1:]}, 60)
        second = compare(data, plain["signal_offset" + key], {"signal_offset_mask": plain["signal_offset_mask" + key]}, 60)
        stress = compare(data, masked["signal_offset" + key + "_mask24_40"],
                         {"signal_offset_mask": masked["signal_offset_mask" + key + "_mask24_40"]}, 60)
        matched_seeds[str(seed)] = {"vs_factored": first["models"], "mask_augmentation": second["models"],
                                    "mask_stress_augmentation": stress["models"]}
    write_json(out / "scores_paired_seeds.json", matched_seeds)
    targets = {row.query_id: row.probabilities for row in read_targets(data / "targets.csv")}
    truth = np.array([targets[key] for key in split.query_ids])
    groups = {r["query_id"]: r["condition"] for r in read_rows(data / "conditions.csv", ("query_id", "condition"))}
    condition = np.array([groups[key] for key in split.query_ids])
    support = (split.X >= 0).sum(axis=1)
    slices = {"all": np.ones(split.n, dtype=bool), "short_history_64": support == 64, "full_prefix": support > 64,
              "mask24_40_eligible_originally_full": eligible_prefixes(split)}
    for val in np.unique(condition):
        slices[f"condition_{val}"] = condition == val
    selected = {"archived_hybrid": HYBRID, "m3": M3, **external, **plain, **masked}

    def aligned(path):
        p = read_submission(path, split.query_ids)
        return np.array([p[key] for key in split.query_ids])

    predictions = {name: aligned(path) for name, path in selected.items()}
    reference = aligned(data / "reference.csv")
    losses = {name: np.square(p - truth) for name, p in predictions.items()}
    reference_sse = np.square(reference - truth).sum(axis=1)
    slice_rows = []
    for slice_name, keep in slices.items():
        if not keep.any():
            continue
        # Recompute a reference-normalized mean over the episodes and conditions represented in the slice.
        episode, index, counts = np.unique(split.episode_ids[keep], return_inverse=True, return_counts=True)
        group_by_episode = np.array([condition[split.episode_ids == key][0] for key in episode])
        by_group = [np.flatnonzero(group_by_episode == group) for group in np.unique(group_by_episode)]

        def episode_means(values):
            return np.bincount(index, weights=values[keep]) / counts

        reference_error = episode_means(reference_sse)
        rng = np.random.default_rng(20261107)
        boot = [rng.choice(group, (10000, len(group))) for group in by_group]
        denom = [reference_error[draw].mean(axis=1).clip(1e-9) for draw in boot]

        def ratio(episode_error):
            return float(np.mean([episode_error[ix].mean() / max(1e-9, reference_error[ix].mean()) for ix in by_group]))

        for name, loss in losses.items():
            if name.endswith("_mask24_40"):
                control_name = "signal_offset_mask24_40"
            elif "hybrid_" in name:
                control_name = "archived_hybrid"
            elif name == "signal_offset_mask":
                control_name = "signal_offset"
            else:
                control_name = "factored"
            error = episode_means(loss.sum(axis=1))
            control_error = episode_means(losses[control_name].sum(axis=1))
            diff = error - control_error
            draws = np.mean([diff[ix].mean(axis=1) / d for ix, d in zip(boot, denom)], axis=0)
            ci = np.quantile(draws, [.025, .975])
            per_class = {f"R_p{payout}": ratio(episode_means(loss[:, i])) for i, payout in enumerate((0, 1, 2, 5, 20))}
            slice_rows.append({"slice": slice_name, "queries": int(keep.sum()), "episodes": len(episode),
                               "model": name, "control": control_name, "R": ratio(error), "delta_R": ratio(diff),
                               "low": float(ci[0]), "high": float(ci[1]),
                               "p0_mse": float(loss[keep, 0].mean()), **per_class})
    write_rows(out / "slices.csv", tuple(slice_rows[0]), slice_rows)

    signal = split.signal
    with np.load(frozen / "signal_counterfactuals.npz", allow_pickle=False) as archive:
        assert np.array_equal(archive["query_ids"], split.query_ids)
        counters = {n: archive[n] for n in archive.files if n != "query_ids"}

    def observed_gap(p0, keep):
        return float(p0[keep & (signal == -1)].mean() - p0[keep & (signal == 1)].mean())

    signal_report = {}
    for label, keep in slices.items():
        if not (np.any(keep & (signal == -1)) and np.any(keep & (signal == 1))):
            continue
        signal_report[label] = {
            "observed_counts": {str(s): int((keep & (signal == s)).sum()) for s in (-1, 0, 1)},
            "target_observed_gap": observed_gap(truth[:, 0], keep),
            "observed_gaps": {n: observed_gap(p[:, 0], keep) for n, p in predictions.items()},
            "same_history_forced_signal_gap": {n: float((arr[0, keep] - arr[1, keep]).mean())
                                               for n, arr in counters.items()},
        }
    write_json(out / "signal_slopes.json", signal_report)
    print(f"E6 frozen scoring finished in {perf_counter() - start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
