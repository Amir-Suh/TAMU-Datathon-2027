"""Independent metamorphic checks for episode exclusion and cache provenance."""
from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.common.crossfit import FitCache
from models.common.data import MISSING, Split
from models.m5_boosted_corrector.model import BoostedCorrector


def small_train() -> Split:
    episodes = 6
    machines = 3
    n = episodes * machines
    cutoff = np.full(n, 40, dtype=int)
    history = np.full((n, 41), MISSING, dtype=np.int8)
    for row in range(n):
        for tick in range(40):
            history[row, tick] = int((tick + row * 3) % 7 in (0, 1))
    return Split(
        name="metamorphic-train",
        query_ids=[f"q{row}" for row in range(n)],
        episode_ids=np.array([f"e{row // machines}" for row in range(n)]),
        machine_ids=[f"m{row % machines}" for row in range(n)],
        machine_type=np.array([row % 3 for row in range(n)]),
        signal=np.array([row % 3 - 1 for row in range(n)]),
        cutoff=cutoff,
        X=history,
        labels=np.array([int(row % 4 == 0) for row in range(n)], dtype=np.int8),
    )


class SensitiveBase:
    name = "sensitive_base"

    def fit(self, train):
        self.p = 0.2 + 0.6 * float(np.mean(train.labels > 0))
        self.size = self
        return self

    def features(self, split):
        n = split.n
        p = np.full(n, self.p)
        zero = np.zeros(n)
        return dict(p_win=p, p_win_independent=p, p_shared=zero,
                    p_null=zero, max_pair_post=np.full(n, 0.5), p_flip=zero,
                    since_flip=zero, n_obs=np.full(n, 40.0),
                    n_recent=np.full(n, 40.0),
                    machine_type=split.machine_type.astype(float),
                    signal=split.signal.astype(float))

    def predict_win(self, split):
        return np.full(split.n, self.p)

    def predict(self, split):
        return np.tile([0.25, 0.25, 0.25, 0.25], (split.n, 1))


class SensitiveClassifier:
    def __init__(self, n_iter):
        self.n_iter = n_iter

    def fit(self, X, y):
        self.training_feature_mean = float(np.mean(X[:, 0]))
        self.training_label_mean = float(np.mean(y))
        return self

    def predict_proba(self, X):
        z = 0.2 * X[:, 0] + 0.3 * self.training_feature_mean + self.training_label_mean
        p = 1 / (1 + np.exp(-z))
        return np.column_stack([1 - p, p])


class ReviewInvarianceTest(unittest.TestCase):
    def test_m5_heldout_forecasts_ignore_heldout_cutoff_labels(self):
        original = small_train()
        changed = replace(original, labels=original.labels.copy())
        changed.labels[changed.episode_ids == "e0"] = 1
        self.assertFalse(np.array_equal(original.labels, changed.labels))

        def fit(train):
            model = BoostedCorrector(
                folds=3, inner_folds=2, base_factory=SensitiveBase,
                classifier_factory=SensitiveClassifier, progress=lambda _: None,
            )
            model.ITER_GRID = (1,)  # fixed choice isolates the heldout fit invariant
            return model.fit(train)

        before, after = fit(original), fit(changed)
        self.assertEqual(before.oof_row_ids, after.oof_row_ids)
        held = np.array([episode == "e0" for episode, _, _ in before.oof_row_ids])
        np.testing.assert_array_equal(before.oof[held], after.oof[held])
        # The mutation is real and affects fits whose training partitions contain e0.
        self.assertTrue(np.any(before.oof[~held] != after.oof[~held]))

    def test_cache_distinguishes_same_episode_ids_with_changed_outcomes_and_history(self):
        original = small_train()
        cache = FitCache()
        first = cache.fit("base", original, SensitiveBase)
        self.assertIs(first, cache.fit("base", original, SensitiveBase))

        labels_changed = replace(original, labels=original.labels.copy())
        labels_changed.labels[:3] = 1
        second = cache.fit("base", labels_changed, SensitiveBase)
        self.assertIsNot(first, second)
        self.assertNotEqual(first.p, second.p)

        history_changed = replace(original, X=original.X.copy())
        history_changed.X[0, 0] = 1 - history_changed.X[0, 0]
        third = cache.fit("base", history_changed, SensitiveBase)
        self.assertIsNot(first, third)
        self.assertIsNot(second, third)


if __name__ == "__main__":
    unittest.main()
