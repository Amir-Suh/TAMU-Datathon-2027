"""Cross-fit a win-logit offset from the frozen factored GRU payout feature."""

import argparse
from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                 "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"

import numpy as np
import scipy
from scipy.stats import spearmanr
from sklearn import __version__ as sklearn_version
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
KIT = HERE.parents[2]
sys.path.insert(0, str(KIT))

from signal_experiments.score import compare, sha256
from slot_forecasting.io import (
    read_queries, read_rows, read_submission, read_targets, require_ids,
    write_json, write_rows, write_submission,
)
from slot_forecasting.schema import CONDITIONS, PAYOUTS

calibration = importlib.import_module("signal_experiments.03_signal_calibration.calibrate")
ceiling = importlib.import_module("signal_experiments.ceiling_analysis.analyze")
EXPERIMENTS = KIT / "signal_experiments"
PUBLIC = KIT / "public"
DATA = PUBLIC / "validation"
BASE = EXPERIMENTS / "combined/m3_match_split_win_pair_l31_factor_gru_size.csv"
DONOR = EXPERIMENTS / "01_fredy_replay/forecasts/gru128_factor_mean_3.csv"
FOLDS = EXPERIMENTS / "folds.csv"
PRIOR_SIGNAL = (EXPERIMENTS / "03_signal_calibration/results/fredy_factored_gru_mean"
                / "official_signal_vs_intercept.json")
WIN_BINS = np.linspace(0, 1, 6)
VARIANTS = ("raw", "intercept", "intercept_state")
EXPECTED_RAW_POINTS = 56.34102223545075


def conditional_payout(p):
    return p[:, 1:] / calibration.check_win(p[:, 1:].sum(axis=1))[:, None]


def fit_fold(raw, feature, types, episode, outcomes, held):
    train = ~held
    standardized = np.empty_like(feature)
    scalers = {}
    for kind in sorted(set(types)):
        rows = types == kind
        values = feature[rows & train]
        center, scale = values.mean(), values.std(ddof=0)
        if not np.isfinite(scale) or scale == 0:
            raise ValueError(f"Invalid fitting-fold scale for {kind}.")
        standardized[rows] = (feature[rows] - center) / scale
        scalers[kind] = {"mean": float(center), "sd": float(scale), "fit_rows": len(values)}

    base_win = raw[train, 1:].sum(axis=1)
    fits = {}
    predictions = {}
    for variant in VARIANTS[1:]:
        coefficients = calibration.fit_offsets(
            base_win, standardized[train], outcomes[train], episode[train],
            with_signal=variant == "intercept_state",
        )
        fits[variant] = coefficients
        predictions[variant] = calibration.reweight_win(raw[held], standardized[held], coefficients)
    return predictions, fits, scalers, standardized[held]


def crossfit(raw, feature, types, cutoffs, fold_ids):
    forecasts = {name: raw.copy() for name in VARIANTS}
    oof_z = np.empty(len(raw))
    details = []
    for fold in sorted(set(fold_ids)):
        held = fold_ids == fold
        predict, fits, scalers, held_z = fit_fold(
            raw, feature, types, cutoffs.episode, cutoffs.win, held,
        )
        changed_labels = cutoffs.win.copy()
        changed_labels[held] = np.where(changed_labels[held] < .5, 1, 0)
        perturbed, altered_fits, altered_scalers, altered_z = fit_fold(
            raw, feature, types, cutoffs.episode, changed_labels, held,
        )
        if fits != altered_fits or scalers != altered_scalers or not np.array_equal(held_z, altered_z):
            raise ValueError(f"Fold {fold} fitting changed after its held cutoff outcomes were flipped.")
        for name, values in predict.items():
            if not np.array_equal(values, perturbed[name]):
                raise ValueError(f"{name} predictions depend on held outcomes in fold {fold}.")
            forecasts[name][held] = values
        oof_z[held] = held_z
        details.append({
            "fold": int(fold),
            "fit_episodes": len(set(cutoffs.episode[~held])),
            "held_episodes": len(set(cutoffs.episode[held])),
            "fit_cutoffs": int((~held).sum()),
            "held_cutoffs": int(held.sum()),
            "fit_scalers": scalers,
            "coefficients": {name: {"intercept": a, "state_slope": b} for name, (a, b) in fits.items()},
            "held_label_flip_bitwise_identical": True,
        })
    return forecasts, details, oof_z


def diagnostic_rows(forecasts, masks, true_win, observed, episodes):
    rows = []
    for group, slices in masks.items():
        for label, mask in slices.items():
            for name, p in forecasts.items():
                win = 1 - p[mask, 0]
                truth = true_win[mask]
                realized = observed[mask]
                rows.append(dict(
                    group=group, slice=label, model=name, queries=int(mask.sum()),
                    episodes=len(set(episodes[mask])), mean_win_forecast=win.mean(),
                    mean_win_target=truth.mean(), mean_win_observed=realized.mean(),
                    win_bias=(win - truth).mean(), p0_mse=np.square(win - truth).mean(),
                    observed_win_brier=np.square(win - realized).mean(),
                    observed_win_log_loss=-np.where(realized > 0, np.log(win), np.log1p(-win)).mean(),
                ))
    return rows


def evaluate_diagnostics(cutoffs, predictions, types, feature, standardized, fold_ids, out):
    # Targets and inferred pair slices are diagnostics only. The forecasts are already written.
    split = ceiling.load_split(PUBLIC, "validation")
    if not np.array_equal(split.query_ids, cutoffs.query_ids):
        raise ValueError("Pair and target diagnostics disagree with cutoff query IDs.")
    true = split.targets
    true_win = 1 - true[:, 0]
    true_q1 = conditional_payout(true)[:, 0]
    mode = np.empty(len(true), dtype=bool)
    mode_results = {}
    raw_win = 1 - predictions["raw"][:, 0]
    bin_id = np.digitize(raw_win, WIN_BINS[1:-1])
    residual = true_win - raw_win
    for kind in sorted(set(types)):
        sel = types == kind
        levels = np.unique(true_q1[sel].round(12))
        if len(levels) != 2:
            raise ValueError(f"{kind}: expected two released 1x modes.")
        mode[sel] = true_q1[sel] > levels.mean()
        residual_rho = spearmanr(feature[sel], residual[sel]).statistic
        mode_results[kind] = {
            "released_conditional_1x_levels": levels.tolist(),
            "donor_mode_auc": float(roc_auc_score(mode[sel], feature[sel])),
            "donor_vs_raw_win_target_residual_spearman": float(residual_rho),
        }

    pair, _, _, _ = ceiling.pair_diagnostics(split)
    focal = {"inferred_pair_4_7": (pair == [4, 7]).all(axis=1),
             "short_history": split.conditions == "short_history"}
    labels = {"low_1x": ~mode, "high_1x": mode}
    masks = {
        "overall": {"all": np.ones(len(true), dtype=bool)},
        "mode": labels,
        "type_mode": {f"{kind}/{level}": (types == kind) & mask
                      for kind in mode_results for level, mask in labels.items()},
        "raw_predicted_win_bin": {
            f"[{lo:.1f},{hi:.1f}{']' if i == 4 else ')'}": bin_id == i
            for i, (lo, hi) in enumerate(zip(WIN_BINS[:-1], WIN_BINS[1:]))
        },
        "focal": focal,
    }
    rows = diagnostic_rows(predictions, masks, true_win, cutoffs.win, cutoffs.episode)
    write_rows(out / "diagnostics.csv", tuple(rows[0]), rows)

    audit_rows = [dict(query_id=key, fold=int(fold), machine_type=kind, gru_conditional_1x=f,
                       fit_standardized_state=z, inferred_pair=f"{lags[0]}-{lags[1]}")
                  for key, fold, kind, f, z, lags in
                  zip(cutoffs.query_ids, fold_ids, types, feature, standardized, pair)]
    write_rows(out / "state_features.csv", tuple(audit_rows[0]), audit_rows)

    reference_loss = np.square(split.reference - true).sum(axis=1)
    official_weights, _ = ceiling.weights(split.episode_ids, split.conditions, reference_loss)
    loss = {name: np.square(p - true).sum(axis=1) for name, p in predictions.items()}
    reference_arms = {"raw": ("intercept", "intercept_state"), "intercept": ("intercept_state",)}
    slice_results = {}
    for baseline, candidates in reference_arms.items():
        slice_results[baseline] = {}
        for label, mask in {**focal, **labels}.items():
            differences = {name: (loss[name] - loss[baseline]) * mask for name in candidates}
            intervals = ceiling.paired_intervals(split, reference_loss, differences, np.zeros(len(true)))
            slice_results[baseline][label] = {
                name: {"delta_R_contribution": float(official_weights @ delta),
                       "delta_R_contribution_95": intervals[name]["delta_R_95"]}
                for name, delta in differences.items()
            }

    # Remove within-type fixed win-bin means before describing incremental feature association.
    # This uses released odds only after prediction and is not an offset fit.
    centered_residual = residual.copy()
    centered_state = standardized.copy()
    for kind in mode_results:
        for index in range(len(WIN_BINS) - 1):
            mask = (types == kind) & (bin_id == index)
            if mask.any():
                centered_residual[mask] -= centered_residual[mask].mean()
                centered_state[mask] -= centered_state[mask].mean()
    return {
        "released_mode_by_type": mode_results,
        "partial_state_residual_correlation_within_type_raw_win_bins": float(
            np.corrcoef(centered_state, centered_residual)[0, 1]),
        "official_delta_R_slice_contributions": slice_results,
    }


def git_metadata():
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=KIT, text=True,
                            capture_output=True).stdout.strip()
    head = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=KIT, text=True,
                          capture_output=True).stdout.strip()
    return {"branch": branch or None, "commit": head or None}


def soft_odds_sensitivity(out, raw, feature, types, cutoffs, folds, raw_path, points):
    """Secondary diagnostic chosen after the primary result. Never fit held odds."""
    targets = {target.query_id: target for target in read_targets(DATA / "targets.csv")}
    require_ids(targets, cutoffs.query_ids)
    win = np.array([1 - targets[key].probabilities[0] for key in cutoffs.query_ids])
    forecasts, fits, _ = crossfit(raw, feature, types, replace(cutoffs, win=win), folds)
    paths = {name: out / f"secondary_soft_{name}.csv" for name in VARIANTS[1:]}
    for name, path in paths.items():
        write_submission(path, dict(zip(cutoffs.query_ids, forecasts[name])))
    comparisons = {
        "vs_raw": compare(DATA, raw_path, paths, points),
        "vs_intercept": compare(DATA, paths["intercept"],
                                {"intercept_state": paths["intercept_state"]}, points),
    }
    return {
        "scope": "Post-hoc grouped development sensitivity. Uses soft win probabilities from "
                 "other validation folds with identical penalties and features.",
        "comparison": comparisons,
        "fold_fits": fits,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    metadata_path = PUBLIC / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    cutoffs = calibration.Cutoffs.read(DATA)
    queries = read_queries(DATA / "queries.csv")
    types = np.array([query.machine_type for query in queries])
    if tuple(q.query_id for q in queries) != cutoffs.query_ids:
        raise ValueError("Machine types disagree with cutoff query order.")
    conditions = {r["query_id"]: r["condition"] for r in
                  read_rows(DATA / "conditions.csv", ("query_id", "condition"))}
    require_ids(conditions, cutoffs.query_ids)
    if set(conditions.values()) != set(CONDITIONS):
        raise ValueError("Unexpected condition set.")
    folds = calibration.validation_folds(FOLDS, cutoffs, np.array([conditions[q] for q in cutoffs.query_ids]))

    raw = cutoffs.predictions(BASE)
    feature = conditional_payout(cutoffs.predictions(DONOR))[:, 0]
    forecasts, fits, standardized = crossfit(raw, feature, types, cutoffs, folds)
    paths = {"raw": out / "raw.csv", "intercept": out / "intercept.csv",
             "intercept_state": out / "validation_predictions.csv"}
    shutil.copyfile(BASE, paths["raw"])
    for name in VARIANTS[1:]:
        write_submission(paths[name], dict(zip(cutoffs.query_ids, forecasts[name])))
    for name, path in paths.items():
        by_query = read_submission(path, cutoffs.query_ids)
        forecasts[name] = np.array([by_query[key] for key in cutoffs.query_ids])
    mix_drift = {name: float(np.max(np.abs(conditional_payout(p) - conditional_payout(raw))))
                 for name, p in forecasts.items()}
    if max(mix_drift.values()) > 1e-14:
        raise ValueError(f"Conditional size changed after export: {mix_drift}")

    points = metadata["max_points"]
    comparisons = {
        "vs_raw": compare(DATA, paths["raw"], {name: paths[name] for name in VARIANTS[1:]}, points),
        "vs_intercept": compare(DATA, paths["intercept"], {"intercept_state": paths["intercept_state"]}, points),
    }
    if abs(comparisons["vs_raw"]["models"]["control"]["points"] - EXPECTED_RAW_POINTS) > 1e-10:
        raise ValueError("The frozen raw forecast did not reproduce the published score.")
    for report in comparisons.values():
        report["data"] = str(DATA.relative_to(KIT))
        for model in report["models"].values():
            model["path"] = Path(model["path"]).name

    negative = json.loads(PRIOR_SIGNAL.read_text())["models"]["intercept_signal"]
    secondary = soft_odds_sensitivity(out, raw, feature, types, cutoffs, folds, paths["raw"], points)
    for report in secondary["comparison"].values():
        report["data"] = str(DATA.relative_to(KIT))
        for model in report["models"].values():
            model["path"] = Path(model["path"]).name
    write_json(out / "scores.json", {
        "comparison": comparisons,
        "diagnostics": evaluate_diagnostics(cutoffs, forecasts, types, feature, standardized, folds, out),
        "prior_signal_negative_comparison": {
            "scope": "Prior cross-fitted signal slope versus intercept on the standalone factored GRU base.",
            "delta_R": negative["delta_R"],
            "delta_R_95": negative["delta_R_95"],
        },
        "export_conditional_mix_max_abs_drift": mix_drift,
        "fold_fits": fits,
        "secondary_post_hoc_soft_odds": secondary,
    })

    files = [BASE, DONOR, FOLDS, PRIOR_SIGNAL, metadata_path, Path(__file__).resolve(),
             Path(calibration.__file__), Path(ceiling.__file__), EXPERIMENTS / "score.py",
             ceiling.FROZEN / "models.joblib", ceiling.FROZEN / "freeze.json"]
    files.extend(DATA / name for name in ("queries.csv", "labels.csv", "history.csv", "targets.csv",
                                         "conditions.csv", "reference.csv"))
    write_json(out / "plan.json", {
        "experiment": "Z1",
        "round": metadata["round"],
        "season_id": metadata["season_id"],
        "git": git_metadata(),
        "scope": "Grouped main-validation development. Frozen parents use train cutoff labels in the GRU "
                 "and transductive pre-cutoff histories. The offset fits only other validation-fold "
                 "actual cutoff outcomes, with no fit on released validation probabilities.",
        "arms": list(VARIANTS),
        "payout_order": list(PAYOUTS),
        "folds": str(FOLDS.relative_to(KIT)),
        "fit": {
            "feature": "Frozen GRU conditional 1x probability, z-scored by type in fitting episodes.",
            "objective": "Bernoulli log loss with equal episode weight and a fixed raw win-logit offset.",
            "intercept_prior_sd": calibration.INTERCEPT_SD,
            "state_slope_prior_sd": calibration.SIGNAL_SD,
            "seed": None,
            "actual_cutoff_outcomes_only": True,
        },
        "bootstrap": {"draws": 10000, "seed": 20261107, "stratified_by": "condition", "unit": "episode"},
        "secondary_post_hoc": "Repeat the two adjusted arms with fitting-fold released win odds.",
        "frozen_inputs_sha256": {str(path.relative_to(KIT)): sha256(path) for path in files},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "scikit_learn": sklearn_version},
        "max_cpu_threads": 2,
    })


if __name__ == "__main__":
    main()
