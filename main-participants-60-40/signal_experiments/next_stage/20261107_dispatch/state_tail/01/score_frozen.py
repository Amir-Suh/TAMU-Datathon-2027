"""Score only immutable E1/E2 validation forecasts."""
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import monotonic

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
KIT = HERE.parents[4]
EXP = KIT / "signal_experiments"
DATA = KIT / "public/validation"
sys.path[:0] = [str(EXP), str(KIT)]
from score import BASELINE, compare
from slot_forecasting.io import read_queries, read_submission, read_targets

ARMS = ("hybrid", "p0", "e1_payout", "e1_decoupled", "e1_coupled", "e2_finite", "e1_coupled_e2_finite")
MATCH = {
    "e1_payout": "hybrid", "e1_decoupled": "e1_payout", "e1_coupled": "e1_decoupled",
    "p0": "hybrid", "e2_finite": "hybrid", "e1_coupled_e2_finite": "e1_coupled",
}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def dump(name, data):
    (HERE / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def ratios(p):
    total = p[:, 2:].sum(axis=1, keepdims=True)
    return np.divide(p[:, 2:], total, out=np.zeros_like(p[:, 2:]), where=total > 0)


def score_slices(paths):
    targets = read_targets(DATA / "targets.csv")
    queries = {q.query_id: q for q in read_queries(DATA / "queries.csv")}
    ids = [t.query_id for t in targets]
    eps = np.array([t.episode_id for t in targets])
    types = np.array([queries[key].machine_type for key in ids])
    truth = np.array([t.probabilities for t in targets])
    truth_ratio = ratios(truth)
    with (HERE / "validation_state.csv").open(newline="") as stream:
        state = {row["query_id"]: row for row in csv.DictReader(stream)}
    posterior = np.array([float(state[key]["gamma"]) for key in ids])
    n_tail = np.array([int(state[key]["n_tail"]) for key in ids])
    state_mid = np.array([(float(state[key]["mu_low"]) + float(state[key]["mu_high"])) / 2 for key in ids])
    truth_high = (truth[:, 1] / truth[:, 1:].sum(axis=1)) > state_mid
    with (DATA / "conditions.csv").open(newline="") as stream:
        group_map = {row["query_id"]: row["condition"] for row in csv.DictReader(stream)}
    condition = np.array([group_map[key] for key in ids])
    with (HERE.parents[3] / "folds.csv").open(newline="") as stream:
        folds = {r["episode_id"]: int(r["fold"]) for r in csv.DictReader(stream) if r["split"] == "validation"}
    rows = {
        "all": np.ones(len(ids), dtype=bool),
        "balanced_or_jackpot": np.isin(types, ["balanced", "jackpot"]),
        "uncertain_state_gamma_0.2_to_0.8": (posterior >= 0.2) & (posterior <= 0.8),
        "balanced_or_jackpot_uncertain": np.isin(types, ["balanced", "jackpot"]) &
                                         (posterior >= 0.2) & (posterior <= 0.8),
        "support_tail_0_to_16": n_tail <= 16,
        "support_tail_17_to_40": (n_tail > 16) & (n_tail <= 40),
        "support_tail_41_plus": n_tail > 40,
    }
    for m in ("steady", "balanced", "jackpot"):
        rows[m] = types == m
        rows[m + "_state_high_from_odds"] = (types == m) & truth_high
        rows[m + "_state_low_from_odds"] = (types == m) & ~truth_high
    for c in sorted(set(condition)):
        rows[c] = condition == c
    for fold in range(4):
        rows[f"main_development_fold_{fold}"] = np.array([folds[e] == fold for e in eps])
    ref = read_submission(DATA / "reference.csv", ids)
    ref_error = np.square(np.asarray([ref[q] for q in ids]) - truth).sum(axis=1)
    _, row_to_episode, ep_sizes = np.unique(eps, return_inverse=True, return_counts=True)
    strata = {}
    rng = np.random.default_rng(20261107)
    for c in sorted(set(condition)):
        e = np.unique(row_to_episode[condition == c])
        draw = rng.choice(e, (10_000, len(e)))
        denominator = (np.bincount(row_to_episode, weights=ref_error) / ep_sizes)[draw].mean(axis=1)
        strata[c] = draw, denominator
    arrays = {}
    for key, path in paths.items():
        p = read_submission(path, ids)
        arrays[key] = np.asarray([p[q] for q in ids])
    comparisons = [(arm, "hybrid") for arm in ARMS if arm != "hybrid"] + [
        ("e1_decoupled", "e1_payout"), ("e1_coupled", "e1_decoupled"),
        ("e2_finite", "p0"), ("e1_coupled_e2_finite", "e1_coupled"),
    ]
    fields = ["candidate", "control", "slice", "queries", "episodes", "delta_five_class_sse",
              "delta_0x_sse", "delta_1x_sse", "delta_2x_sse", "delta_5x_sse", "delta_20x_sse",
              "delta_tail_sse", "delta_tail_ratio_sse", "candidate_mean_win_bias", "control_mean_win_bias",
              "candidate_mean_ratio_2x", "candidate_mean_ratio_5x", "candidate_mean_ratio_20x",
              "control_mean_ratio_2x", "control_mean_ratio_5x", "control_mean_ratio_20x",
              "true_mean_ratio_2x", "true_mean_ratio_5x", "true_mean_ratio_20x",
              "slice_delta_R_contribution_paired95_low", "slice_delta_R_contribution_paired95_high"]
    with (HERE / "slices.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for candidate, control in comparisons:
            a, b = arrays[candidate], arrays[control]
            a_ratio, b_ratio = ratios(a), ratios(b)
            diff = np.square(a - truth) - np.square(b - truth)
            tail_diff = np.square(a_ratio - truth_ratio) - np.square(b_ratio - truth_ratio)
            for name, mask in rows.items():
                if not mask.any():
                    continue
                class_diff = diff[mask].mean(axis=0)
                ep_diff = np.bincount(row_to_episode, weights=diff.sum(axis=1) * mask) / ep_sizes
                boot = np.mean([ep_diff[draw].mean(axis=1) / denom for draw, denom in strata.values()], axis=0)
                writer.writerow([
                    candidate, control, name, int(mask.sum()), len(set(eps[mask])),
                    class_diff.sum(), *class_diff, class_diff[2:].sum(), tail_diff[mask].sum(axis=1).mean(),
                    (truth[mask, 0] - a[mask, 0]).mean(), (truth[mask, 0] - b[mask, 0]).mean(),
                    *a_ratio[mask].mean(axis=0), *b_ratio[mask].mean(axis=0), *truth_ratio[mask].mean(axis=0),
                    *np.quantile(boot, [0.025, 0.975]),
                ])


def main():
    start = monotonic()
    frozen = json.loads((HERE / "frozen.json").read_text())
    assert frozen["phase"] == "forecasts_frozen_before_scoring"
    actual = {name: digest(HERE / name) for name in frozen["sha256"]}
    if actual != frozen["sha256"]:
        raise ValueError("A forecast changed after freeze")
    paths = {arm: HERE / f"validation_{arm}.csv" for arm in ARMS}
    hybrid = paths["hybrid"]
    hybrid_report = compare(DATA, hybrid, {k: p for k, p in paths.items() if k != "hybrid"}, 60)
    m3_report = compare(DATA, BASELINE, paths, 60)
    matched = {}
    for candidate, control in MATCH.items():
        matched[candidate] = compare(DATA, paths[control], {candidate: paths[candidate]}, 60)
    matched["e2_finite_vs_p0"] = compare(DATA, paths["p0"], {"e2_finite": paths["e2_finite"]}, 60)
    dump("scores_vs_hybrid.json", hybrid_report)
    dump("scores_vs_m3.json", m3_report)
    dump("scores_vs_matched.json", matched)
    score_slices(paths)
    dump("score_phase.json", {
        "phase": "scored_frozen_forecasts", "forecast_freeze_sha256": digest(HERE / "frozen.json"),
        "score_end_utc": datetime.now(timezone.utc).isoformat(), "score_seconds": monotonic() - start,
        "fit_files_unchanged": actual == {name: digest(HERE / name) for name in actual},
    })
    for name, values in hybrid_report["models"].items():
        print(f"{name:23s} {values['points']:.9f} R={values['R']:.9f} "
              f"delta={values['delta_R']:+.9f} {values['delta_R_95']}")


if __name__ == "__main__":
    main()
