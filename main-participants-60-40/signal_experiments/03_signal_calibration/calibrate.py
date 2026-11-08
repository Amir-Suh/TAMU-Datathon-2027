"""Fit and evaluate cutoff-only win-logit offsets for frozen forecasts."""

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

KIT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KIT))

from slot_forecasting.io import (
    read_labels,
    read_queries,
    read_rows,
    read_submission,
    read_targets,
    require_ids,
    write_json,
    write_rows,
    write_submission,
)

INTERCEPT_SD = 1.0
SIGNAL_SD = 0.5
VARIANTS = ("raw", "intercept", "intercept_signal")
CONTRASTS = (("intercept", "raw"), ("intercept_signal", "raw"), ("intercept_signal", "intercept"))


@dataclass(frozen=True)
class Cutoffs:
    query_ids: tuple
    episode: np.ndarray
    signal: np.ndarray
    win: np.ndarray

    @classmethod
    def read(cls, directory):
        queries = read_queries(directory / "queries.csv")
        labels = read_labels(directory / "labels.csv", queries)
        return cls(
            tuple(q.query_id for q in queries),
            np.array([q.episode_id for q in queries]),
            np.array([q.signal for q in queries]),
            np.array([labels[q.query_id] > 0 for q in queries], dtype=float),
        )

    def predictions(self, path):
        prediction = read_submission(path, self.query_ids)
        return np.array([prediction[key] for key in self.query_ids])


def episode_weights(episodes):
    counts = Counter(episodes)
    return np.array([1.0 / counts[episode] for episode in episodes])


def fit_offsets(base_win, signal, outcome, episodes, *, with_signal):
    """Penalized Bernoulli log likelihood with the base logit as a fixed offset."""
    weights = episode_weights(episodes)
    # Each episode has equal influence and weights sum to the number of labels.
    weights *= len(weights) / weights.sum()
    base_logit = logit(base_win)
    design = np.ones((len(base_logit), 2 if with_signal else 1))
    prior_sd = np.array([INTERCEPT_SD, SIGNAL_SD] if with_signal else [INTERCEPT_SD])
    if with_signal:
        design[:, 1] = signal

    def objective(coef):
        eta = base_logit + design @ coef
        penalty = np.square(coef / prior_sd).sum() / 2
        loss = weights @ (np.logaddexp(0, eta) - outcome * eta) + penalty
        gradient = design.T @ (weights * (expit(eta) - outcome)) + coef / prior_sd**2
        return loss, gradient

    fit = minimize(objective, np.zeros(design.shape[1]), jac=True, method="L-BFGS-B",
                   options={"ftol": 1e-14, "gtol": 1e-7})
    if not fit.success:
        raise RuntimeError(f"Cutoff calibration failed: {fit.message}")
    return (float(fit.x[0]), float(fit.x[1]) if with_signal else 0.0)


def check_win(win):
    if not np.all(np.isfinite(win) & (win > 0) & (win < 1)):
        raise ValueError("Base win probabilities must be strictly between zero and one.")
    return win


def reweight_win(probabilities, signal, offsets):
    """Change total positive mass while preserving the four-way conditional mix."""
    base_win = check_win(probabilities[:, 1:].sum(axis=1))
    intercept, slope = offsets
    win = expit(logit(base_win) + intercept + slope * signal)
    conditional_size = probabilities[:, 1:] / base_win[:, None]
    return np.column_stack((1 - win, conditional_size * win[:, None]))


def archived_train_win(path, name, cutoffs):
    """Read a verified OOF cache containing endpoint and pseudo-query rows."""
    with np.load(path, allow_pickle=False) as cache:
        row_ids = cache["row_ids"]
        mask = cache["is_endpoint"]
        if not np.array_equal(mask, row_ids[:, 2] == "0"):
            raise ValueError("Archived cutoff markers and cutoff shifts disagree.")
        endpoint_ids = row_ids[mask, 1]
        index = {query: i for i, query in enumerate(endpoint_ids)}
        if len(index) != len(endpoint_ids):
            raise ValueError("Duplicate archived train cutoff IDs.")
        require_ids(index, cutoffs.query_ids)
        order = np.array([index[key] for key in cutoffs.query_ids])
        if not (np.array_equal(row_ids[mask, 0][order], cutoffs.episode)
                and np.array_equal(cache["y"][mask][order], cutoffs.win)):
            raise ValueError("Archived cutoff episodes or outcomes disagree with the kit.")
        return check_win(cache[name][mask][order])


def validation_folds(path, cutoffs, conditions):
    rows = [row for row in read_rows(path, ("split", "episode_id", "fold"))
            if row["split"] == "validation"]
    folds = {row["episode_id"]: int(row["fold"]) for row in rows}
    if len(folds) != len(rows):
        raise ValueError("Duplicate validation fold assignments.")
    require_ids(folds, cutoffs.episode)
    fold_id = np.array([folds[episode] for episode in cutoffs.episode])
    if len(set(fold_id)) != 4:
        raise ValueError("Expected four validation folds.")
    for condition in set(conditions):
        counts = Counter(folds[episode] for episode in set(cutoffs.episode[conditions == condition]))
        if len(counts) != 4 or max(counts.values()) - min(counts.values()) > 1:
            raise ValueError(f"Validation folds are imbalanced for {condition}.")
    return fold_id


def fit_variants(validation, raw, *, folds=None, train=None, train_win=None):
    if (folds is None) == (train is None):
        raise ValueError("Choose either independent training cutoffs or validation cross-fitting.")
    forecasts = {"raw": raw}
    coefficient_rows = []
    if folds is None:
        fit_data, fit_win = train, check_win(train_win)
        partitions = [("held_validation", np.ones(len(raw), dtype=bool), np.ones(len(train_win), dtype=bool))]
    else:
        fit_data, fit_win = validation, check_win(raw[:, 1:].sum(axis=1))
        partitions = [(str(fold), folds == fold, folds != fold) for fold in sorted(set(folds))]

    for variant in VARIANTS[1:]:
        adjusted = np.empty_like(raw)
        for fold, held, use in partitions:
            offset = fit_offsets(fit_win[use], fit_data.signal[use], fit_data.win[use],
                                 fit_data.episode[use], with_signal=(variant == "intercept_signal"))
            adjusted[held] = reweight_win(raw[held], validation.signal[held], offset)
            coefficient_rows.append(dict(
                variant=variant, fold=fold, fit_episodes=len(set(fit_data.episode[use])),
                fit_cutoff_labels=int(use.sum()), held_episodes=len(set(validation.episode[held])),
                intercept=offset[0], signal_slope=offset[1], odds_multiplier_per_signal_step=float(np.exp(offset[1])),
            ))
        forecasts[variant] = adjusted
    return forecasts, coefficient_rows


def diagnostics(validation, forecasts, conditions, target_win, repeats=4000):
    """Observed-label metrics, paired episode deltas and descriptive signal residuals."""
    episodes, episode_index = np.unique(validation.episode, return_inverse=True)
    row_weights = episode_weights(validation.episode)
    groups = np.array([conditions[np.flatnonzero(episode_index == i)[0]] for i in range(len(episodes))])
    rng = np.random.default_rng(314159)
    draws = np.concatenate([
        rng.choice(indices, size=(repeats, len(indices)), replace=True)
        for group in sorted(set(groups))
        for indices in [np.flatnonzero(groups == group)]
    ], axis=1)
    metrics, episode_scores, signal_rows = {}, {}, []
    for variant, pred in forecasts.items():
        win = pred[:, 1:].sum(axis=1)
        row_scores = {
            "observed_win_brier": np.square(win - validation.win),
            "observed_win_logloss": np.where(validation.win > 0, -np.log(win), -np.log1p(-win)),
        }
        episode_scores[variant] = {
            name: np.bincount(episode_index, weights=loss * row_weights)
            for name, loss in row_scores.items()
        }
        metrics[variant] = {name: float(loss.mean()) for name, loss in episode_scores[variant].items()}
        for signal in (-1, 0, 1):
            select = validation.signal == signal
            signal_rows.append(dict(
                variant=variant, signal=signal, queries=int(select.sum()),
                mean_forecast_win=float(win[select].mean()),
                mean_observed_win=float(validation.win[select].mean()),
                mean_target_win=float(target_win[select].mean()),
                mean_observed_residual=float((validation.win[select] - win[select]).mean()),
                mean_target_residual=float((target_win[select] - win[select]).mean()),
            ))
    contrasts = {}
    for contender, baseline in CONTRASTS:
        contrasts[f"{contender}_vs_{baseline}"] = {}
        for metric, loss in episode_scores[contender].items():
            delta = loss - episode_scores[baseline][metric]
            contrasts[f"{contender}_vs_{baseline}"][metric] = {
                "delta": float(delta.mean()),
                "paired_ci95": np.quantile(delta[draws].mean(axis=1), [0.025, 0.975]).tolist(),
            }
    return metrics, contrasts, signal_rows


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    directory = args.public / "validation"
    validation = Cutoffs.read(directory)
    raw = validation.predictions(args.validation_forecast)
    conditions_rows = read_rows(directory / "conditions.csv", ("query_id", "condition"))
    conditions_by_id = {row["query_id"]: row["condition"] for row in conditions_rows}
    if len(conditions_by_id) != len(conditions_rows):
        raise ValueError("Duplicate validation condition assignment.")
    require_ids(conditions_by_id, validation.query_ids)
    conditions = np.array([conditions_by_id[key] for key in validation.query_ids])
    inputs = [args.validation_forecast, directory / "queries.csv", directory / "labels.csv",
              directory / "conditions.csv", directory / "targets.csv"]
    if args.train_forecast is not None or args.train_cache is not None:
        train_dir = args.public / "train"
        train = Cutoffs.read(train_dir)
        if set(train.episode) & set(validation.episode):
            raise ValueError("Training and validation episode IDs overlap.")
        if args.train_forecast is not None:
            train_win = train.predictions(args.train_forecast)[:, 1:].sum(axis=1)
            inputs.append(args.train_forecast)
            scope = "Train cutoff calibration on fixed base predictions independent of all train cutoff outcomes, evaluated on held validation."
        else:
            train_win = archived_train_win(args.train_cache, args.train_cache_key, train)
            inputs.append(args.train_cache)
            scope = "Train OOF cutoff calibration, fixed before validation scoring."
        forecasts, coefficients = fit_variants(validation, raw, train=train, train_win=train_win)
        inputs.extend([train_dir / "queries.csv", train_dir / "labels.csv"])
    else:
        folds = validation_folds(args.folds, validation, conditions)
        forecasts, coefficients = fit_variants(validation, raw, folds=folds)
        scope = "Four-fold validation cutoff cross-fitting by whole episode, development only."
        inputs.append(args.folds)

    # Simulator probabilities enter only after the fit and prediction steps.
    targets = {target.query_id: target for target in read_targets(directory / "targets.csv")}
    require_ids(targets, validation.query_ids)
    target_win = 1 - np.array([targets[key].probabilities[0] for key in validation.query_ids])
    for i, key in enumerate(validation.query_ids):
        if targets[key].episode_id != validation.episode[i] or (targets[key].outcome > 0) != validation.win[i]:
            raise ValueError("Validation targets disagree with query episode or observed win.")
    metrics, contrasts, signal_rows = diagnostics(validation, forecasts, conditions, target_win)
    original_mix = raw[:, 1:] / raw[:, 1:].sum(axis=1, keepdims=True)
    mix_error = {
        name: float(np.max(np.abs(p[:, 1:] / p[:, 1:].sum(axis=1, keepdims=True) - original_mix)))
        for name, p in forecasts.items()
    }
    args.out.mkdir(parents=True, exist_ok=False)
    for name, forecast in forecasts.items():
        write_submission(args.out / f"{name}.csv", dict(zip(validation.query_ids, forecast)))
    write_rows(args.out / "coefficients.csv", tuple(coefficients[0]), coefficients)
    write_rows(args.out / "signal_residuals.csv", tuple(signal_rows[0]), signal_rows)
    write_json(args.out / "calibration.json", {
        "model": args.model,
        "fit_scope": scope,
        "prior_sd": {"intercept": INTERCEPT_SD, "signal_slope": SIGNAL_SD},
        "target_probabilities_used_in_fit": False,
        "validation_episodes": len(set(validation.episode)),
        "validation_queries": len(validation.query_ids),
        "input_sha256": {str(path): sha256(path) for path in inputs},
        "conditional_mix_max_change": mix_error,
        "metrics": metrics,
        "contrasts": contrasts,
        "bootstrap_scope": "Whole episodes sampled within condition, with OOF forecasts fixed.",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Explicit provenance label for the base model")
    parser.add_argument("--public", type=Path, default=KIT / "public")
    parser.add_argument("--validation-forecast", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--folds", type=Path, help="Validation grouped OOF fold manifest")
    mode.add_argument("--train-forecast", type=Path, help="Standard CSV of train forecasts independent of all train cutoff outcomes")
    mode.add_argument("--train-cache", type=Path, help="Archived repaired M3 OOF cutoff NPZ")
    parser.add_argument("--train-cache-key", default="m3")
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
