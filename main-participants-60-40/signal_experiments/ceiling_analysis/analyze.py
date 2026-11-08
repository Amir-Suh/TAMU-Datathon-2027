"""Locate remaining main-validation forecast error using frozen predictions."""

import csv
from dataclasses import replace
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
KIT = HERE.parents[1]
EXPERIMENTS = HERE.parent
REPAIR = KIT / "m5_m6_repair"
FROZEN = REPAIR / "results/repaired-m5-m6"
sys.path[:0] = [str(KIT), str(REPAIR), str(EXPERIMENTS)]

from models.common.data import CONDITIONS, MISSING, PAYOUTS, TYPES, load_split
from score import sha256
from slot_forecasting.io import read_submission


CANDIDATES = {
    "reference": KIT / "public/validation/reference.csv",
    "m1_frequency": KIT / "adversarial_review/branches/amir/results/2026-11-07_m1_to_m6/m1_hierarchical_frequency/validation_predictions.csv",
    "m2_lag_pair": KIT / "adversarial_review/branches/amir/results/2026-11-07_m1_to_m6/m2_lag_pair/validation_predictions.csv",
    **{name: FROZEN / name / "validation_predictions.csv" for name in ("m3", "m4", "m5", "m6")},
    "pair_gbdt": EXPERIMENTS / "01_fredy_replay/forecasts/pair_l31.csv",
    "plain_gru": EXPERIMENTS / "01_fredy_replay/forecasts/gru128_nomsg_mean.csv",
    "factored_gru": EXPERIMENTS / "01_fredy_replay/forecasts/gru128_factor_mean_3.csv",
    "full_blend": EXPERIMENTS / "01_fredy_replay/forecasts/pair_l31_factor_gru_m6_repaired.csv",
    "payout_state_m3": EXPERIMENTS / "02_payout_state/validation_predictions.csv",
    "split_match": EXPERIMENTS / "04_four_lag_cells/validation_predictions.csv",
    "tree_hybrid": EXPERIMENTS / "combined/m3_match_split_win_pair_l31_size.csv",
    "plain_gru_hybrid": EXPERIMENTS / "combined/m3_match_split_win_pair_l31_gru_size.csv",
    "best": EXPERIMENTS / "combined/m3_match_split_win_pair_l31_factor_gru_size.csv",
}


def payout_mix(p):
    win = p[:, 1:].sum(axis=1)
    return p[:, 1:] / win[:, None]


def combine(win, size):
    return np.column_stack((1 - win, win[:, None] * size))


def pair_diagnostics(split):
    checkpoint = FROZEN / "models.joblib"
    manifest = json.loads((FROZEN / "freeze.json").read_text())
    if sha256(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("Unexpected M3 checkpoint.")
    model = joblib.load(checkpoint)["m3"]
    causal = replace(split, targets=None, labels=None, reference=None, conditions=None)
    post = model._infer(causal)["ind"]["post"]
    pair_post = post.sum(axis=(2, 3))
    pair = np.asarray(model.pairs)[pair_post.argmax(axis=1)]
    observed = split.X != MISSING
    ticks = np.arange(split.X.shape[1])[None, :]
    a, b = ticks - pair[:, :1], ticks - pair[:, 1:]
    support = observed & (b >= 0) & np.take_along_axis(observed, a.clip(0), axis=1)
    support &= np.take_along_axis(observed, b.clip(0), axis=1)
    window = (model.tau_grid[None, :] >= split.cutoff[:, None] - 40)
    recent_flip = (post[:, :, :, 1:].sum(axis=(1, 2)) * window).sum(axis=1)
    return pair, pair_post.max(axis=1), support.sum(axis=1), recent_flip


def weights(episode, condition, reference_loss):
    episode_keys, indices, counts = np.unique(episode, return_inverse=True, return_counts=True)
    per_row = np.zeros(len(episode))
    denominators = {}
    for name in CONDITIONS:
        mask = condition == name
        num_episodes = len(np.unique(indices[mask]))
        episode_weight = 1 / (num_episodes * counts[indices[mask]])
        denom = float(episode_weight @ reference_loss[mask])
        per_row[mask] = episode_weight / (len(CONDITIONS) * denom)
        denominators[name] = denom
    if np.any(per_row == 0) or len(episode_keys) != sum(len(set(episode[condition == c])) for c in CONDITIONS):
        raise ValueError("Conditions must partition episodes.")
    return per_row, denominators


def masks_for(split, pair, confidence, support, recent_flip, true_win, mode_high):
    def labeled(values, labels):
        return {label: values == i for i, label in enumerate(labels)}

    pair_mask = {f"{a}-{b}": (pair == [a, b]).all(axis=1) for a, b in np.unique(pair, axis=0)}
    return {
        "condition": {c: split.conditions == c for c in CONDITIONS},
        "history": {"64": (split.X != MISSING).sum(axis=1) == 64,
                    "full": (split.X != MISSING).sum(axis=1) == split.cutoff},
        "pair_support": labeled(np.digitize(support, [40, 100, 200]), ["<40", "40–99", "100–199", "200+"]),
        "pair_posterior": labeled(np.digitize(confidence, [.9, .99]), ["<.90", ".90–.99", "≥.99"]),
        "pair_identity": pair_mask,
        "pair_by_condition": {
            f"{name} / {condition}": mask & (split.conditions == condition)
            for name, mask in pair_mask.items() for condition in CONDITIONS
        },
        "pair_by_type": {
            f"{name} / {kind}": mask & (split.machine_type == i)
            for name, mask in pair_mask.items() for i, kind in enumerate(TYPES)
        },
        "recent_flip_posterior": labeled(np.digitize(recent_flip, [.1, .5]), ["<.10", ".10–.49", "≥.50"]),
        "cutoff": labeled(np.digitize(split.cutoff, [214, 236]), ["192–213", "214–235", "236–256"]),
        "machine_type": labeled(split.machine_type, TYPES),
        "signal": labeled(split.signal + 1, ["-1", "0", "+1"]),
        "true_win_probability": labeled(np.digitize(true_win, [.1, .5, .9]), ["<.10", ".10–.49", ".50–.89", "≥.90"]),
        "true_payout_mode": {"low 1x": ~mode_high, "high 1x": mode_high},
    }


def paired_intervals(val, reference, candidates, baseline):
    _, index, counts = np.unique(val.episode_ids, return_inverse=True, return_counts=True)
    groups = {name: np.unique(index[val.conditions == name]) for name in CONDITIONS}
    rng = np.random.default_rng(20261107)
    draws = {name: rng.choice(group, (10000, len(group))) for name, group in groups.items()}
    ref_episode = np.bincount(index, weights=reference) / counts
    denominators = {name: ref_episode[sample].mean(axis=1) for name, sample in draws.items()}
    result = {}
    for name, losses in candidates.items():
        diff = np.bincount(index, weights=losses - baseline) / counts
        ratios = {c: diff[sample].mean(axis=1) / denominators[c] for c, sample in draws.items()}
        result[name] = {
            "condition_delta_R_95": {c: np.quantile(v, [.025, .975]).tolist() for c, v in ratios.items()},
            "delta_R_95": np.quantile(np.mean(list(ratios.values()), axis=0), [.025, .975]).tolist(),
        }
    return result


def main():
    train = load_split(KIT / "public", "train")
    val = load_split(KIT / "public", "validation")
    pred = {}
    for name, path in CANDIDATES.items():
        by_id = read_submission(path, val.query_ids)
        pred[name] = np.asarray([by_id[key] for key in val.query_ids])
    true = val.targets
    true_mix, true_win = payout_mix(true), 1 - true[:, 0]
    mode_high = np.zeros(val.n, dtype=bool)
    for i in range(len(TYPES)):
        rows = val.machine_type == i
        levels = np.unique(true_mix[rows, 0].round(12))
        if len(levels) != 2:
            raise ValueError("Expected two conditional 1x levels per machine type.")
        mode_high[rows] = true_mix[rows, 0] > levels.mean()

    # Fit only the tail ratios from training histories. Keep the leading forecast's
    # win probability and conditional 1x probability fixed for this probe.
    train_tail = np.array([[np.count_nonzero(train.X[train.machine_type == i] == p)
                            for p in range(2, 5)] for i in range(len(TYPES))], dtype=float)
    train_tail /= train_tail.sum(axis=1, keepdims=True)
    best_mix = payout_mix(pred["best"])
    projected_mix = np.column_stack((best_mix[:, 0], (1 - best_mix[:, :1]) * train_tail[val.machine_type]))
    pred["train_tail_projection"] = combine(1 - pred["best"][:, 0], projected_mix)

    pair, confidence, support, recent_flip = pair_diagnostics(val)
    per_class = {name: np.square(p - true) for name, p in pred.items()}
    errors = {name: e.sum(axis=1) for name, e in per_class.items()}
    w, denominators = weights(val.episode_ids, val.conditions, errors["reference"])
    model_results = {}
    for name, prediction in pred.items():
        win, mix = 1 - prediction[:, 0], payout_mix(prediction)
        win_oracle_error = np.square(combine(true_win, mix) - true).sum(axis=1)
        size_oracle_error = np.square(combine(win, true_mix) - true).sum(axis=1)
        loss = errors[name]
        r = float(w @ loss)
        win_r = float(w @ (loss - win_oracle_error + size_oracle_error) / 2)
        size_r = r - win_r
        model_results[name] = {
            "points": 60 / (1 + r), "R": r,
            "condition_R": {c: float(w[val.conditions == c] @ loss[val.conditions == c] * 4) for c in CONDITIONS},
            "shapley_R": {"win": win_r, "size": size_r},
            "oracle_win_points": float(60 / (1 + w @ win_oracle_error)),
            "oracle_size_points": float(60 / (1 + w @ size_oracle_error)),
            "oracle_win_condition_R": {c: float(w[val.conditions == c] @ win_oracle_error[val.conditions == c] * 4) for c in CONDITIONS},
            "oracle_size_condition_R": {c: float(w[val.conditions == c] @ size_oracle_error[val.conditions == c] * 4) for c in CONDITIONS},
            "mse_by_class": dict(zip(PAYOUTS, per_class[name].mean(axis=0).tolist())),
            "R_by_class": dict(zip(PAYOUTS, (w @ per_class[name]).tolist())),
            "win_weighted_conditional_size_SSE": float(np.mean(true_win**2 * np.square(mix - true_mix).sum(axis=1))),
            "mean_SSE": float(loss.mean()),
            "conditional_1x_auc_by_type": {
                kind: float(roc_auc_score(mode_high[val.machine_type == i], mix[val.machine_type == i, 0]))
                for i, kind in enumerate(TYPES)
            },
        }

    if abs(model_results["best"]["points"] - 56.34102223545075) > 1e-10:
        raise ValueError("Common scorer replay disagrees with the saved best model.")
    best_R = model_results["best"]["R"]
    slices = {}
    for grouping, groups in masks_for(val, pair, confidence, support, recent_flip, true_win, mode_high).items():
        slices[grouping] = {}
        for label, mask in groups.items():
            if not np.any(mask):
                continue
            res = {
                "queries": int(mask.sum()),
                "episodes": len(set(val.episode_ids[mask])),
                "mean_true_win": float(true_win[mask].mean()),
                "median_observed_pair_support": float(np.median(support[mask])),
                "best_residual_R_share": float(w[mask] @ errors["best"][mask] / best_R),
                "best_gain_vs_m3_in_R": float(w[mask] @ (errors["m3"][mask] - errors["best"][mask])),
                "best_p0_R": float(w[mask] @ per_class["best"][mask, 0]),
                "best_p1_R": float(w[mask] @ per_class["best"][mask, 1]),
                "mean_SSE": {name: float(values[mask].mean()) for name, values in errors.items()},
                "best_mean_win_bias": float(np.mean(1 - pred["best"][mask, 0] - true_win[mask])),
            }
            slices[grouping][label] = res

    # Concentration ranks rows by their contribution to the official loss ratio.
    contribution = w * errors["best"]
    order = np.argsort(-contribution)
    hardest = order[:round(.1 * val.n)]
    report = {
        "scope": "Fixed forecast analysis on reused main validation, 96 episodes and 768 queries.",
        "denominators": denominators,
        "data_sha256": {name: sha256(KIT / "public/validation" / name) for name in
                        ("queries.csv", "history.csv", "targets.csv", "conditions.csv", "reference.csv")},
        "forecast_sha256": {name: sha256(path) for name, path in CANDIDATES.items()},
        "models": model_results,
        "slices": slices,
        "paired_vs_best": paired_intervals(
            val, errors["reference"],
            {name: errors[name] for name in ("m3", "full_blend", "plain_gru_hybrid", "train_tail_projection")},
            errors["best"],
        ),
        "top_tenth": {"queries": len(hardest), "R_share": float(contribution[hardest].sum() / best_R),
                      "conditions": {c: int(np.count_nonzero(val.conditions[hardest] == c)) for c in CONDITIONS},
                      "machine_types": {kind: int(np.count_nonzero(val.machine_type[hardest] == i)) for i, kind in enumerate(TYPES)}},
        "train_history_tail_ratio": dict(zip(TYPES, train_tail.tolist())),
        "validation_target_tail_range_by_type": {
            kind: np.ptp(true_mix[val.machine_type == i, 1:] /
                         true_mix[val.machine_type == i, 1:].sum(axis=1, keepdims=True), axis=0).tolist()
            for i, kind in enumerate(TYPES)
        },
    }
    (HERE / "diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")
    with (HERE / "slices.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["group", "slice", "queries", "episodes", "true_win", "median_pair_support",
                         "best_R_share", "R_gain_vs_m3", "best_win_bias", *[f"SSE_{name}" for name in pred]])
        for group, groups in slices.items():
            for label, row in groups.items():
                writer.writerow([group, label, row["queries"], row["episodes"], row["mean_true_win"],
                                 row["median_observed_pair_support"], row["best_residual_R_share"],
                                 row["best_gain_vs_m3_in_R"], row["best_mean_win_bias"], *row["mean_SSE"].values()])


if __name__ == "__main__":
    main()
