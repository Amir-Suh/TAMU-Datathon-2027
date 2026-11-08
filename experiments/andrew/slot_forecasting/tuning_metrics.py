"""Paired uncertainty estimates for the official episode-weighted loss ratio."""

from collections import defaultdict

import numpy as np

from .io import require_ids
from .schema import CONDITIONS, ValidationError, integer
from .scoring import distribution_error


def paired_ratio_interval(candidate, control, targets, groups, reference, *, repeats=2000, seed=314159):
    """Resample episodes within condition, including the reference denominator."""
    repeats = integer(repeats, "bootstrap repeats", 1)
    seed = integer(seed, "bootstrap seed")
    targets = tuple(targets)
    ids = {target.query_id for target in targets}
    if len(ids) != len(targets):
        raise ValidationError("Expected unique tuning targets.")
    for forecasts in (candidate, control, groups, reference):
        require_ids(forecasts, ids)
    if set(groups.values()) != set(CONDITIONS):
        raise ValidationError("Expected all four tuning conditions.")

    by_episode = defaultdict(list)
    episode_conditions = {}
    for target in sorted(targets, key=lambda row: row.query_id):
        key = target.query_id
        episode = target.episode_id
        condition = groups[key]
        if episode_conditions.setdefault(episode, condition) != condition:
            raise ValidationError("An episode cannot span multiple tuning conditions.")
        truth = target.probabilities
        difference = distribution_error(candidate[key], truth) - distribution_error(control[key], truth)
        by_episode[episode].append((difference, distribution_error(reference[key], truth)))

    rng = np.random.default_rng(seed)
    draws = np.zeros(repeats)
    delta = 0.0
    for condition in CONDITIONS:
        episodes = [np.mean(by_episode[episode], axis=0) for episode in sorted(by_episode)
                    if episode_conditions[episode] == condition]
        episodes = np.asarray(episodes)
        difference, denominator = episodes.mean(axis=0)
        delta += difference / max(denominator, 1e-9)
        sample = rng.integers(len(episodes), size=(repeats, len(episodes)))
        means = episodes[sample].mean(axis=1)
        draws += means[:, 0] / np.maximum(means[:, 1], 1e-9)
    lower, upper = np.quantile(draws / len(CONDITIONS), [0.025, 0.975])
    return {"delta_r": float(delta / len(CONDITIONS)), "lower": float(lower), "upper": float(upper),
            "repeats": repeats, "seed": seed}
