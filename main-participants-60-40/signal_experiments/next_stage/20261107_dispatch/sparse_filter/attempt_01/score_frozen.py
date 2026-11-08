"""Verify the E4 freeze and score forecasts and paired slices against released odds."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parents[1]
EXP = RUN.parents[1]
KIT = EXP.parent
sys.path[:0] = [str(KIT), str(EXP)]

from score import BASELINE, compare, sha256
from slot_forecasting.io import read_queries, read_submission, read_targets, write_json, write_rows
from slot_forecasting.schema import CONDITIONS, PAYOUTS

DATA = KIT / "public/validation"
HYBRID = EXP / "combined/m3_match_split_win_pair_l31_factor_gru_size.csv"
S1 = EXP / "research_next/S1/validation_global.csv"
M4 = BASELINE.parent.parent / "m4/validation_predictions.csv"
PAIR_SLICE = RUN / "lag_cells/attempt_01/frozen_validation_slices.csv"
SEED = 20261107
DRAWS = 10_000
CONTROLS = {"forward": "drop", "constant": "forward", "gated": "constant"}
ARM_NAMES = ("drop", "forward", "constant", "gated", "hybrid", "m3", "m4", "donor")


def load_rows(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def paired_slices(tiers):
    targets = read_targets(DATA / "targets.csv")
    ids = [target.query_id for target in targets]
    truth = np.asarray([target.probabilities for target in targets])
    episodes = np.array([target.episode_id for target in targets])
    condition_map = {row["query_id"]: row["condition"] for row in load_rows(DATA / "conditions.csv")}
    conditions = np.asarray([condition_map[q] for q in ids])
    support = {row["query_id"]: float(row["support"]) for row in load_rows(HERE / "validation_support_diagnostics.csv")
               if row["budget"] == "released"}
    pair = {row["query_id"]: row["top_pair"] for row in load_rows(PAIR_SLICE)}

    def array(path):
        mapping = read_submission(path, ids)
        return np.asarray([mapping[q] for q in ids])

    reference_sq = (array(DATA / "reference.csv") - truth)**2
    hybrid_sq = (array(HYBRID) - truth)**2
    selected_slices = {
        "overall": np.ones(len(ids), dtype=bool), "released_short_history": conditions == "short_history",
        "released_complete_history": conditions != "short_history",
        "released_forward_support_below_40": np.asarray([support[q] < 40 for q in ids]),
        "frozen_m3_top_pair_4_7": np.asarray([pair[q] == "4-7" for q in ids]),
    }
    output = []
    for (budget, tier), paths in tiers.items():
        p = {name: array(path) for name, path in paths.items()}
        sq = {name: (value - truth)**2 for name, value in p.items()}
        sq["released_frozen_hybrid"] = hybrid_sq
        sq["reference"] = reference_sq
        for slice_name, selected in selected_slices.items():
            rng = np.random.default_rng(SEED)
            ratios, draws = defaultdict(list), defaultdict(list)
            for condition in CONDITIONS:
                rows = selected & (conditions == condition)
                if not rows.any():
                    continue
                _, inverse, counts = np.unique(episodes[rows], return_inverse=True, return_counts=True)
                losses = {name: np.bincount(inverse, weights=values[rows].sum(axis=1))/counts
                          for name, values in sq.items()}
                draw_index = rng.integers(len(counts), size=(DRAWS, len(counts)))
                reference_mean = np.clip(losses["reference"].mean(), 1e-9, None)
                sampled_reference = losses["reference"][draw_index].mean(axis=1).clip(1e-9)
                for name, values in losses.items():
                    ratios[name].append(values.mean()/reference_mean)
                    draws[name].append(values[draw_index].mean(axis=1)/sampled_reference)
            r = {name: np.mean(value) for name, value in ratios.items()}
            d = {name: np.mean(value, axis=0) for name, value in draws.items()}
            q_true = truth[:, 1:]/truth[:, 1:].sum(axis=1, keepdims=True)
            for name, value in p.items():
                control = CONTROLS.get(name, "hybrid")
                hybrid_ci = np.quantile(d[name]-d["released_frozen_hybrid"], [.025, .975])
                mask_hybrid_ci = np.quantile(d[name]-d["hybrid"], [.025, .975])
                matched_ci = np.quantile(d[name]-d[control], [.025, .975])
                q = value[:, 1:]/value[:, 1:].sum(axis=1, keepdims=True)
                payout_sse = ((1 - truth[:, 0])**2 * np.sum((q_true-q)**2, axis=1))[selected].mean()
                classes = sq[name][selected].mean(axis=0)
                output.append({"budget": budget, "tier": tier, "slice": slice_name, "arm": name,
                               "queries": int(selected.sum()), "episodes": len(set(episodes[selected])),
                               "R": r[name], "points": 60/(1+r[name]), "matched_control": control,
                               "delta_R_vs_matched": r[name]-r[control],
                               "matched_ci_low": matched_ci[0], "matched_ci_high": matched_ci[1],
                               "delta_R_vs_released_hybrid": r[name]-r["released_frozen_hybrid"],
                               "released_hybrid_ci_low": hybrid_ci[0], "released_hybrid_ci_high": hybrid_ci[1],
                               "delta_R_vs_same_mask_hybrid": r[name]-r["hybrid"],
                               "same_mask_hybrid_ci_low": mask_hybrid_ci[0],
                               "same_mask_hybrid_ci_high": mask_hybrid_ci[1],
                               **{f"p{payout}_mse": error for payout, error in zip(PAYOUTS, classes)},
                               "win_mass_weighted_conditional_payout_sse": payout_sse})
    write_rows(HERE / "slices.csv", tuple(output[0]), output)


def main():
    freeze = json.loads((HERE / "forecast_freeze.json").read_text())
    assert not freeze["smoke"]
    for name, expected in freeze["output_sha256"].items():
        assert sha256(HERE / name) == expected, name
    manifest = freeze["result"]["validation"]["forecasts"]
    entries = {name: HERE / filename for name, filename in manifest.items()}
    archived = {"archived_M3": BASELINE, "archived_M4": M4, "archived_S1_global": S1}
    assert len(read_queries(DATA / "queries.csv")) == 768
    vs_hybrid = compare(DATA, HYBRID, {**entries, **archived}, 60, DRAWS)
    write_json(HERE / "scores_vs_hybrid.json", vs_hybrid)
    write_json(HERE / "scores_vs_m3.json", compare(DATA, BASELINE, entries, 60, DRAWS))

    tiers = {}
    for key, path in entries.items():
        budget, tier, arm = key.split("/")
        tiers.setdefault((budget, tier), {})[arm] = path
    matched = {}
    for (budget, tier), paths in tiers.items():
        assert set(paths) == set(ARM_NAMES)
        name = f"{budget}/{tier}"
        matched[name] = {
            "forward_vs_drop": compare(DATA, paths["drop"], {"forward": paths["forward"]}, 60, DRAWS),
            "constant_vs_forward": compare(DATA, paths["forward"], {"constant": paths["constant"]}, 60, DRAWS),
            "gated_vs_constant": compare(DATA, paths["constant"], {"gated": paths["gated"]}, 60, DRAWS),
            "all_vs_same_mask_hybrid": compare(DATA, paths["hybrid"],
                                                {k: v for k, v in paths.items() if k != "hybrid"}, 60, DRAWS),
        }
    write_json(HERE / "scores_vs_matched.json", matched)
    paired_slices(tiers)
    for name, row in vs_hybrid["models"].items():
        print(f"{name:42} {row['points']:.6f}  delta_R={row['delta_R']:+.7f}  "
              f"short_R={row['condition_R']['short_history']:.7f}", flush=True)


if __name__ == "__main__":
    main()
