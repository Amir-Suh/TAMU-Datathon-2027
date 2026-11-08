"""Split M3's predictive match cell into 00 and 11, with frozen evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
KIT = HERE.parents[1]
REPAIR = KIT / "m5_m6_repair"
FROZEN = REPAIR / "results/repaired-m5-m6"
sys.path[:0] = [str(REPAIR), str(KIT)]

from models.common.components import combine
from models.common.crossfit import subset_split
from models.common.data import MISSING, Split
from models.m3_lag_pair_flip.model import FlipLagPairModel, _lagged
from run_repaired_comparison import input_only
from slot_forecasting.io import read_submission, read_targets, require_ids, write_json, write_submission
from slot_forecasting.rounds import score_round

GLOBAL_PRIOR = 200.0
LOCAL_PRIOR = 20.0
MIN_HISTORY_PSEUDO = 8.0


@dataclass
class MatchCells:
    n00: np.ndarray
    wins00: np.ndarray
    n11: np.ndarray
    wins11: np.ndarray
    current: np.ndarray  # -1 unknown, 0 = 00, 1 = 11, 2 = differ

    def evidence(self) -> tuple[np.ndarray, np.ndarray]:
        n = self.n00 + self.n11
        effective = self.n00 * self.n11 / np.maximum(n, 1)
        observed = (self.wins11 / np.maximum(self.n11, 1)
                    - self.wins00 / np.maximum(self.n00, 1))
        return effective, observed


def read_checkpoint(path: Path) -> FlipLagPairModel:
    manifest = json.loads((path / "freeze.json").read_text())
    checkpoint = path / "models.joblib"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != manifest["checkpoint_sha256"]:
        raise ValueError("Frozen M3 checkpoint hash differs from its manifest.")
    model = joblib.load(checkpoint)["m3"]
    if type(model) is not FlipLagPairModel or model.share_episode:
        raise ValueError("Expected independent frozen M3.")
    return model


def win_history(split: Split) -> np.ndarray:
    observed = split.X != MISSING
    if np.any(observed & (np.arange(split.X.shape[1]) >= split.cutoff[:, None])):
        raise ValueError("Found history at or after cutoff.")
    return np.where(observed, split.X > 0, MISSING).astype(np.int8)


def match_cells(w: np.ndarray, cutoff: np.ndarray, pair: tuple[int, int],
                tau_grid: np.ndarray) -> MatchCells:
    a, b = pair
    lag_a, lag_b = _lagged(w, a), _lagged(w, b)
    # Prefix counts at tick g are counts for t < g. Subtraction leaves
    # observed outcomes at t >= g, all of which precede the query cutoff.
    positions = np.r_[0, np.minimum(tau_grid, w.shape[1])]
    won = w == 1
    suffix = []
    for child in (0, 1):
        in_cell = (w >= 0) & (lag_a == child) & (lag_b == child)
        for counted in (in_cell, in_cell & won):
            prefix = np.column_stack([np.zeros(w.shape[0]), counted.cumsum(axis=1)])
            suffix.append(prefix[:, -1:] - prefix[:, positions])
    rows = np.arange(w.shape[0])
    current_a, current_b = lag_a[rows, cutoff], lag_b[rows, cutoff]
    current = np.where(
        (current_a == MISSING) | (current_b == MISSING), MISSING,
        np.where(current_a == current_b, current_a, 2),
    )
    return MatchCells(*suffix, current)


def train_contrasts(model: FlipLagPairModel, train: Split) -> dict:
    w = win_history(train)
    post = model._infer(train)["ind"]["post"]
    n = np.zeros(2)
    difference = np.zeros(2)
    for pair_index, pair in enumerate(model.pairs):
        cells = match_cells(w, train.cutoff, pair, model.tau_grid)
        effective, observed = cells.evidence()
        weights = post[:, pair_index] * effective[:, None, :]
        n += weights.sum(axis=(0, 2))
        difference += (weights * observed[:, None, :]).sum(axis=(0, 2))
    return {
        "direction_order": ["match_high", "match_low"],
        "effective_paired_count": n.tolist(),
        "raw_contrast_11_minus_00": (difference / np.maximum(n, 1)).tolist(),
        "learned_contrast_11_minus_00": (difference / (n + GLOBAL_PRIOR)).tolist(),
        "global_zero_contrast_pseudocount": GLOBAL_PRIOR,
        "local_contrast_pseudocount": LOCAL_PRIOR,
        "local_history_pseudocount": MIN_HISTORY_PSEUDO,
    }


def child_forecasts(parent: np.ndarray, cells: MatchCells,
                    prior: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    effective, observed = cells.evidence()
    n = cells.n00 + cells.n11
    contrast = (LOCAL_PRIOR * prior[None, :, None] + (effective * observed)[:, None, :])
    contrast /= (LOCAL_PRIOR + effective)[:, None, :]
    contrast *= (n / (n + MIN_HISTORY_PSEUDO))[:, None, :]

    # Keep the expected child probabilities centered on M3's match parent
    # under the empirical subtype frequencies (one pseudocount per child).
    frac11 = ((cells.n11 + 1) / (n + 2))[:, None, :]
    frac00 = 1 - frac11
    lower = np.maximum(-(1 - parent) / frac11, -parent / frac00)
    upper = np.minimum(parent / frac11, (1 - parent) / frac00)
    contrast = np.clip(contrast, lower, upper)
    return parent - frac11 * contrast, parent + frac00 * contrast


def predict(model: FlipLagPairModel, split: Split, prior: np.ndarray) -> dict:
    w = win_history(split)
    inference = model._infer(split)
    posterior = inference["ind"]["post"]
    top = posterior.sum(axis=(2, 3)).argmax(axis=1)
    category = np.full(split.n, MISSING)
    delta = np.zeros(split.n)
    match_mass = np.zeros(split.n)
    max_center_error = 0.0
    for pi, pair in enumerate(model.pairs):
        cells = match_cells(w, split.cutoff, pair, model.tau_grid)
        category[top == pi] = cells.current[top == pi]
        parent = inference["core"]["pred"][:, pi]
        match = (cells.current == 0) | (cells.current == 1)
        if not np.any(match):
            continue
        child00, child11 = child_forecasts(parent, cells, prior)
        current = np.where((cells.current == 0)[:, None, None], child00, child11)
        delta[match] += (posterior[match, pi] * (current[match] - parent[match])).sum(axis=(1, 2))
        match_mass[match] += posterior[match, pi].sum(axis=(1, 2))
        frac11 = ((cells.n11 + 1) / (cells.n00 + cells.n11 + 2))[:, None, :]
        center_error = np.abs((1 - frac11) * child00 + frac11 * child11 - parent)
        max_center_error = max(max_center_error, float(center_error.max()))
    frozen_win = inference["p_win"]
    if np.any(delta[match_mass == 0] != 0):
        raise ValueError("Win probability changed without a matching hypothesis.")
    size = model.size.predict(split)
    return {
        "frozen": combine(frozen_win, size),
        "candidate": combine(frozen_win + delta, size),
        "top_pair_category": category,
        "match_posterior_mass": match_mass,
        "win_delta": delta,
        "max_center_error": max_center_error,
    }


def validate_sparse_and_causal(model: FlipLagPairModel, validation: Split,
                               prior: np.ndarray) -> dict:
    observed_counts = (validation.X != MISSING).sum(axis=1)
    rows = np.array([observed_counts.argmin(), observed_counts.argmax()])
    sample = subset_split(validation, rows)
    w = win_history(sample)
    subset_grid = model.tau_grid[::17]
    brute_checks = 0
    for a, b in model.pairs[::7]:
        cells = match_cells(w, sample.cutoff, (a, b), subset_grid)
        for r, cut in enumerate(sample.cutoff):
            for tau_index, tau in enumerate(np.r_[0, subset_grid]):
                for subtype, n, k in ((0, cells.n00, cells.wins00), (1, cells.n11, cells.wins11)):
                    valid = [t for t in range(max(b, int(tau)), cut)
                             if w[r, t] >= 0 and w[r, t - a] == w[r, t - b] == subtype]
                    if n[r, tau_index] != len(valid) or k[r, tau_index] != sum(w[r, t] for t in valid):
                        raise ValueError("Suffix count disagrees with strict pre-cutoff enumeration.")
                    brute_checks += 1
    missing = replace(sample, X=np.full_like(sample.X, MISSING))
    absent = predict(model, missing, prior)
    if np.any(absent["win_delta"] != 0) or np.any(absent["top_pair_category"] != MISSING):
        raise ValueError("Missing lags caused a correction.")
    cells = match_cells(win_history(missing), missing.cutoff, model.pairs[0], subset_grid)
    parent = np.full((sample.n, 2, len(subset_grid) + 1), 0.3)
    child00, child11 = child_forecasts(parent, cells, prior)
    if not np.array_equal(child00, parent) or not np.array_equal(child11, parent):
        raise ValueError("Empty child history failed to return the parent.")
    bad = replace(sample, X=sample.X.copy())
    bad.X[0, bad.cutoff[0]] = 1
    try:
        win_history(bad)
    except ValueError:
        pass
    else:
        raise ValueError("Expected rejection of a payout at cutoff.")
    return {
        "strict_cutoff_brute_force_child_count_checks": brute_checks,
        "all_missing_lag_history_unchanged": True,
        "empty_child_counts_return_exact_parent": True,
        "observed_payout_at_cutoff_rejected": True,
    }


def conditional(p: np.ndarray) -> np.ndarray:
    return p[:, 1:] / p[:, 1:].sum(axis=1, keepdims=True)


def save_and_check(
    split: Split, prediction: dict, frozen_path: Path, out_path: Path
) -> tuple[np.ndarray, np.ndarray, dict]:
    expected = read_submission(frozen_path, split.query_ids)
    base = np.array([expected[q] for q in split.query_ids])
    replay_diff = float(np.abs(base - prediction["frozen"]).max())
    if replay_diff > 1e-12:
        raise ValueError(f"Frozen M3 replay diverged by {replay_diff}.")
    candidate = {q: tuple(p) for q, p in zip(split.query_ids, prediction["candidate"])}
    write_submission(out_path, candidate)
    replayed = read_submission(out_path, split.query_ids)
    values = np.array([replayed[q] for q in split.query_ids])
    size_diff = float(np.abs(conditional(base) - conditional(values)).max())
    if size_diff > 1e-12:
        raise ValueError(f"M3 conditional payout sizes changed by {size_diff}.")
    return base, values, {
        "frozen_m3_replay_max_abs_delta": replay_diff,
        "conditional_payout_max_abs_delta": size_diff,
        "parent_weighted_center_max_abs_delta": prediction["max_center_error"],
        "zero_match_posterior_has_zero_win_delta": True,
    }


def validation_diagnostics(target: np.ndarray, forecast: dict, base: np.ndarray,
                           candidate: np.ndarray) -> dict:
    p0_diff = (candidate[:, 0] - target[:, 0]) ** 2 - (base[:, 0] - target[:, 0]) ** 2
    full_diff = ((candidate - target) ** 2 - (base - target) ** 2).sum(axis=1)
    rows = {}
    category = forecast["top_pair_category"]
    for name, mask in {
        "match": (category == 0) | (category == 1),
        "differ": category == 2,
        "unknown": category == MISSING,
        "00": category == 0,
        "11": category == 1,
    }.items():
        rows[name] = {
            "queries": int(mask.sum()),
            "mean_match_posterior_mass": float(forecast["match_posterior_mass"][mask].mean()),
            "mean_abs_win_delta": float(np.abs(forecast["win_delta"][mask]).mean()),
            "mean_signed_win_delta": float(forecast["win_delta"][mask].mean()),
            "p0_mse_delta": float(p0_diff[mask].mean()),
            "five_class_sse_delta": float(full_diff[mask].mean()),
        } if np.any(mask) else {"queries": 0}
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=KIT / "public")
    parser.add_argument("--frozen", type=Path, default=FROZEN)
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--test", action="store_true", help="Also create an unscored main test CSV.")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        model = read_checkpoint(args.frozen)
        train = input_only(args.data, "train")
        fit = train_contrasts(model, train)
        args.out.mkdir(parents=True, exist_ok=True)
        write_json(args.out / "train_fit.json", fit)

        prior = np.array(fit["learned_contrast_11_minus_00"])
        validation = input_only(args.data, "validation")
        checks = validate_sparse_and_causal(model, validation, prior)
        forecast = predict(model, validation, prior)
        out_csv = args.out / "validation_predictions.csv"
        base_csv = args.frozen / "m3/validation_predictions.csv"
        base, candidate, forecast_checks = save_and_check(validation, forecast, base_csv, out_csv)
        checks.update(forecast_checks)
        test_checks = None
        if args.test:
            test = input_only(args.data, "test")
            _, _, test_checks = save_and_check(
                test, predict(model, test, prior), args.frozen / "m3/submission.csv",
                args.out / "test_predictions.csv",
            )
            test_checks["queries"] = test.n

        # Released validation targets are first read after predictions are written.
        targets = {t.query_id: t.probabilities for t in read_targets(args.data / "validation/targets.csv")}
        require_ids(targets, validation.query_ids)
        target = np.array([targets[q] for q in validation.query_ids])
        base_score = score_round(base_csv, args.data)
        candidate_score = score_round(out_csv, args.data)
        report = {
            "checks": checks,
            "current_top_posterior_pair_context": validation_diagnostics(
                target, forecast, base, candidate
            ),
            "frozen_m3_score": base_score,
            "candidate_score": candidate_score,
        }
        if test_checks is not None:
            report["test_forecast_checks"] = test_checks
        write_json(args.out / "results.json", report)
        print(f"M3 {base_score['technical_points']:.6f}")
        print(f"Match split {candidate_score['technical_points']:.6f}")


if __name__ == "__main__":
    main()
