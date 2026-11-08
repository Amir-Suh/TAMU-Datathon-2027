"""Fit and evaluate a history-only conditional payout-state head on public data."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from m5_m6_repair.models.common.data import MISSING, TYPES, Split, load_split
from slot_forecasting.io import read_rows, read_submission, require_ids, write_json, write_rows
from slot_forecasting.rounds import score_round
from slot_forecasting.schema import PROBABILITY_COLUMNS, SUBMISSION_COLUMNS

M3 = ROOT / "m5_m6_repair/results/repaired-m5-m6/m3"
WINDOWS = (16, 40, 96, math.inf)
PSEUDO_WINS = 8
RIDGE_C = 0.1


@dataclass
class PayoutTemplates:
    """Two conditional 1x levels and one 2x:5x:20x ratio per machine type."""

    levels: np.ndarray
    tail: np.ndarray
    high_rate: np.ndarray

    @classmethod
    def fit(cls, machine_type: np.ndarray, truth: np.ndarray) -> PayoutTemplates:
        conditional = truth[:, 1:] / truth[:, 1:].sum(axis=1, keepdims=True)
        levels, tail, high_rate = [], [], []
        for type_id, name in enumerate(TYPES):
            rows = conditional[machine_type == type_id]
            q = rows[:, 0]
            modes = np.unique(q.round(12))
            if len(modes) != 2:
                raise ValueError(f"{name}: expected two distinct 1x levels, got {modes}")
            is_high = q > modes.mean()
            levels.append([q[~is_high].mean(), q[is_high].mean()])
            tail_ratios = rows[:, 1:] / rows[:, 1:].sum(axis=1, keepdims=True)
            ratios = tail_ratios.mean(axis=0)
            if not np.allclose(tail_ratios, ratios, rtol=0, atol=1e-10):
                raise ValueError(f"{name}: 2x:5x:20x ratio varies in fold-training rows")
            tail.append(ratios / ratios.sum())
            high_rate.append(is_high.mean())
        return cls(np.array(levels), np.array(tail), np.array(high_rate))

    def state(self, machine_type: np.ndarray, truth: np.ndarray) -> np.ndarray:
        q = truth[:, 1] / truth[:, 1:].sum(axis=1)
        return q > self.levels[machine_type].mean(axis=1)

    def sizes(self, machine_type: np.ndarray, prob_high: np.ndarray) -> np.ndarray:
        levels = self.levels[machine_type]
        q = levels[:, 0] + prob_high * (levels[:, 1] - levels[:, 0])
        return np.column_stack([q, (1 - q[:, None]) * self.tail[machine_type]])

    def describe(self) -> dict:
        return {
            name: {
                "p_1_given_win_low": float(self.levels[i, 0]),
                "p_1_given_win_high": float(self.levels[i, 1]),
                "tail_2_5_20": self.tail[i].tolist(),
                "high_state_frequency": float(self.high_rate[i]),
            }
            for i, name in enumerate(TYPES)
        }


def history_features(split: Split, templates: PayoutTemplates) -> np.ndarray:
    """Strictly pre-cutoff own-machine history. Missing ticks contribute no counts."""
    payouts = split.X
    age = split.cutoff[:, None] - np.arange(payouts.shape[1])
    if np.any((payouts != MISSING) & (age <= 0)):
        raise ValueError("History includes a payout at or after the query cutoff.")
    levels = templates.levels[split.machine_type]
    midpoint = levels.mean(axis=1)
    width = levels[:, 1] - levels[:, 0]
    features = [np.eye(len(TYPES))[split.machine_type]]
    for window in WINDOWS:
        included = (age > 0) & (age <= window)
        observed = ((payouts != MISSING) & included).sum(axis=1)
        wins = ((payouts > 0) & included).sum(axis=1)
        ones = ((payouts == 1) & included).sum(axis=1)
        shrunk_ones = (ones - midpoint * wins) / (width * (wins + PSEUDO_WINS))
        shrunk_win_rate = (wins + 2) / (observed + 8)
        features.extend([shrunk_ones, np.log1p(wins), shrunk_win_rate])
    return np.column_stack(features)


def fit_predict(
    source: Split,
    training: np.ndarray,
    destination: Split,
    held: np.ndarray,
    training_weights: np.ndarray | None = None,
) -> tuple[PayoutTemplates, np.ndarray]:
    if set(source.episode_ids[training]) & set(destination.episode_ids[held]):
        raise ValueError("Model fitting and prediction episodes overlap.")
    train_types = source.machine_type[training]
    train_truth = source.targets[training]
    templates = PayoutTemplates.fit(train_types, train_truth)
    train_features = history_features(source, templates)[training]
    held_features = history_features(destination, templates)[held]
    scaler = StandardScaler().fit(train_features)
    model = LogisticRegression(C=RIDGE_C, solver="lbfgs", max_iter=500)
    model.fit(scaler.transform(train_features), templates.state(train_types, train_truth),
              sample_weight=training_weights)
    prob_high = model.predict_proba(scaler.transform(held_features))[:, 1]
    return templates, prob_high


def load_folds(path: Path, validation: Split) -> np.ndarray:
    rows = read_rows(path, ("split", "episode_id", "fold"))
    folds = {}
    for row in rows:
        if row["split"] != "validation":
            continue
        episode = row["episode_id"]
        if episode in folds:
            raise ValueError(f"Duplicate validation fold row for {episode}")
        folds[episode] = int(row["fold"])
    require_ids(folds, validation.episode_ids)
    if len(set(folds.values())) != 4:
        raise ValueError("Expected four validation folds.")
    return np.array([folds[episode] for episode in validation.episode_ids])


def save_forecast(path: Path, base_path: Path, split: Split, sizes: np.ndarray) -> dict:
    """Copy the raw p_0 and preserve it exactly after the kit's normalization."""
    if sizes.shape != (split.n, 4) or not np.allclose(
        sizes.sum(axis=1), 1, rtol=0, atol=1e-13
    ):
        raise ValueError("Invalid conditional payout probabilities.")
    base = read_submission(base_path, split.query_ids)
    raw = {row["query_id"]: row for row in read_rows(base_path, SUBMISSION_COLUMNS)}
    rows = []
    for query, sizes_row in zip(split.query_ids, sizes):
        frozen = [float(raw[query][name]) for name in PROBABILITY_COLUMNS]
        raw_total = math.fsum(frozen)
        positives = list((raw_total - frozen[0]) * sizes_row)
        # Compensate for last-bit floating-point changes so the scorer uses the
        # same normalization denominator as the frozen M3 CSV.
        pivot = int(np.argmax(positives))
        other = [frozen[0], *positives[:pivot], *positives[pivot + 1:]]
        positives[pivot] = raw_total - math.fsum(other)
        total = math.fsum([frozen[0], *positives])
        if total != raw_total:
            direction = math.inf if total < raw_total else -math.inf
            positives[pivot] = math.nextafter(positives[pivot], direction)
        p0 = raw[query]["p_0"]
        rows.append(dict(zip(SUBMISSION_COLUMNS, [query, p0, *positives])))
    write_rows(path, SUBMISSION_COLUMNS, rows)
    forecast = read_submission(path, split.query_ids)
    if any(forecast[query][0] != base[query][0] for query in split.query_ids):
        raise ValueError("Official parsing changed p_0 relative to M3.")
    return {
        "queries": len(rows),
        "p_0_csv_text_copied": True,
        "p_0_after_official_normalization_equal": True,
    }


def validation_experiment(args: argparse.Namespace, validation: Split) -> None:
    folds = load_folds(args.folds, validation)
    sizes = np.empty((validation.n, 4))
    constant_sizes = np.empty_like(sizes)
    weighted_sizes = np.empty_like(sizes)
    scores = np.empty(validation.n)
    constant_scores = np.empty(validation.n)
    weighted_scores = np.empty(validation.n)
    state = np.empty(validation.n, dtype=bool)
    details = {}
    frozen = read_submission(args.m3_validation, validation.query_ids)
    win_squared = np.array([(1 - frozen[q][0]) ** 2 for q in validation.query_ids])
    for fold in np.unique(folds):
        held = folds == fold
        fit, high = fit_predict(validation, ~held, validation, held)
        machine_type = validation.machine_type[held]
        sizes[held] = fit.sizes(machine_type, high)
        constant_scores[held] = fit.high_rate[machine_type]
        constant_sizes[held] = fit.sizes(machine_type, constant_scores[held])
        scores[held] = high
        weights = win_squared[~held]
        weighted_fit, weighted_high = fit_predict(
            validation, ~held, validation, held, weights / weights.mean()
        )
        weighted_sizes[held] = weighted_fit.sizes(machine_type, weighted_high)
        weighted_scores[held] = weighted_high
        # Held soft vectors are used only after inference, for diagnostics.
        state[held] = fit.state(machine_type, validation.targets[held])
        details[str(fold)] = {
            "training_episodes": len(set(validation.episode_ids[~held])),
            "held_episodes": len(set(validation.episode_ids[held])),
            "templates": fit.describe(),
        }
    forecasts = {
        "m3": args.m3_validation,
        "constant_mixture": args.out / "constant_mixture.csv",
        "payout_state": args.out / "validation_predictions.csv",
        "weighted_secondary": args.out / "weighted_secondary.csv",
    }
    probability_checks = {
        "payout_state": save_forecast(
            forecasts["payout_state"], forecasts["m3"], validation, sizes
        ),
        "constant_mixture": save_forecast(
            forecasts["constant_mixture"], forecasts["m3"], validation, constant_sizes
        ),
        "weighted_secondary": save_forecast(
            forecasts["weighted_secondary"], forecasts["m3"], validation, weighted_sizes
        ),
    }
    report = {
        "evaluation": "4-fold episode OOF on reused public validation",
        "feature_windows_in_ticks": [16, 40, 96, "all"],
        "pseudo_wins": PSEUDO_WINS,
        "ridge_c": RIDGE_C,
        "folds": details,
        "probability_checks": probability_checks,
        "oof_high_state_auc": roc_auc_score(state, scores),
        "oof_high_state_brier": brier_score_loss(state, scores),
        "oof_constant_state_brier": brier_score_loss(state, constant_scores),
        "secondary_weights": "Frozen M3 p(win)^2, mean-normalized per training fold",
        "oof_weighted_secondary_state_auc": roc_auc_score(state, weighted_scores),
        "official_scores": {name: score_round(path, args.data) for name, path in forecasts.items()},
    }
    if args.m3_test:
        test = load_split(args.data, "test")
        fit, high = fit_predict(validation, np.ones(validation.n, dtype=bool),
                                test, np.ones(test.n, dtype=bool))
        report["test_forecast"] = save_forecast(
            args.out / "test_predictions.csv", args.m3_test, test,
            fit.sizes(test.machine_type, high)
        )
        report["test_forecast"]["templates_fitted_on_all_public_validation"] = fit.describe()
    write_json(args.out / "results.json", report)
    for name, score in report["official_scores"].items():
        print(f"{name:20s} {score['technical_points']:.6f}  R={score['mean_loss_ratio']:.6f}")
    print(f"OOF state AUC: {report['oof_high_state_auc']:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "public")
    parser.add_argument("--folds", type=Path, default=HERE.parent / "folds.csv")
    parser.add_argument("--m3-validation", type=Path, default=M3 / "validation_predictions.csv")
    parser.add_argument("--m3-test", type=Path)
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        validation_experiment(args, load_split(args.data, "validation"))


if __name__ == "__main__":
    main()
