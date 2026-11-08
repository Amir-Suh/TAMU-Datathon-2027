"""Factorized probability combinations with the contest's quadratic blend loss."""

import csv
import hashlib
import io

import numpy as np

from .io import read_submission
from .schema import CONDITIONS, SUBMISSION_COLUMNS, ValidationError


def load_component(path, ordered_ids):
    content = path.read_bytes()
    header = next(csv.reader(io.StringIO(content.decode("utf-8-sig"))), None)
    if tuple(header or ()) != SUBMISSION_COLUMNS:
        raise ValidationError(f"{path}: expected the submission columns in official order.")
    rows = read_submission(path, ordered_ids, content=content)
    return np.array([rows[key] for key in ordered_ids]), hashlib.sha256(content).hexdigest()


def conditional_positive(forecast, fallback):
    positive = forecast[:, 1:]
    mass = positive.sum(axis=1, keepdims=True)
    return np.divide(positive, mass, out=np.broadcast_to(fallback, positive.shape).copy(), where=mass > 0)


def with_positive_head(no_payout, positive):
    return np.column_stack([no_payout, (1 - no_payout[:, None]) * positive])


def convex_mix(base, alternative, weight):
    return base + weight * (alternative - base)


def official_row_weights(truth, reference, episodes, conditions):
    """One weight per query, with equally weighted episodes within each condition."""
    reference_errors = np.square(reference - truth).sum(axis=1)
    weights = np.zeros(len(episodes))
    for condition in CONDITIONS:
        mask = conditions == condition
        _, row_episode, query_counts = np.unique(episodes[mask], return_inverse=True, return_counts=True)
        if not len(query_counts):
            raise ValidationError("A fitting fold must include all conditions.")
        episode_errors = np.bincount(row_episode, weights=reference_errors[mask]) / query_counts
        denominator = max(1e-9, episode_errors.mean())
        weights[mask] = 1 / (len(CONDITIONS) * denominator * len(query_counts) * query_counts[row_episode])
    if np.any(weights == 0):
        raise ValidationError("Unknown condition in fitting fold.")
    return weights


def fitted_weight(base, alternative, truth, weights):
    direction = alternative - base
    denominator = weights @ np.square(direction).sum(axis=1)
    if denominator <= 1e-15:
        return 0.0
    numerator = weights @ (direction * (truth - base)).sum(axis=1)
    return float(np.clip(numerator / denominator, 0, 1))


def crossfit_blend(base, alternative, truth, reference, episodes, conditions, folds):
    forecast = np.empty_like(base)
    seen = np.zeros(len(episodes), dtype=bool)
    fold_weights = []
    for held_out in folds:
        held = np.isin(episodes, sorted(held_out))
        if not held.any() or np.any(seen & held):
            raise ValidationError("Every tuning query must appear in exactly one held-out fold.")
        fit = ~held
        if set(episodes[fit]) & set(episodes[held]):
            raise ValidationError("Tuning folds must be separated by episode.")
        weights = official_row_weights(truth[fit], reference[fit], episodes[fit], conditions[fit])
        weight = fitted_weight(base[fit], alternative[fit], truth[fit], weights)
        forecast[held] = convex_mix(base[held], alternative[held], weight)
        seen |= held
        fold_weights.append(weight)
    if not np.all(seen):
        raise ValidationError("The tuning folds did not cover all queries.")
    all_weights = official_row_weights(truth, reference, episodes, conditions)
    final_weight = fitted_weight(base, alternative, truth, all_weights)
    return forecast, fold_weights, final_weight
