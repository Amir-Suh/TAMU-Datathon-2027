"""Fast episode-exclusion and alignment tests for the stacked models."""
from __future__ import annotations

import pickle
import unittest

import numpy as np

from models.common.crossfit import FitCache, episode_folds, pseudo_frame, subset_split
from models.common.data import Split
from models.m5_boosted_corrector import BoostedCorrector
from models.m6_ensemble import Ensemble


def sample_train(n_episodes=16):
    n = n_episodes * 2
    cutoff = np.full(n, 36)
    X = np.zeros((n, 37), dtype=np.int8)
    for i in range(n):
        X[i, :36] = ((np.arange(36) + i * 3) % 7 == 0).astype(np.int8)
        X[i, 36] = -1
    return Split(name="train", query_ids=[f"q{i}" for i in range(n)],
                 episode_ids=np.asarray([f"e{i // 2:02d}" for i in range(n)]),
                 machine_ids=[f"m{i}" for i in range(n)], machine_type=np.zeros(n, dtype=int),
                 signal=np.zeros(n, dtype=int), cutoff=cutoff, X=X,
                 labels=np.asarray([(i * 5) % 7 == 0 for i in range(n)], dtype=np.int8))


class FakeBase:
    name = "fake_base"
    feature_calls = []

    def __init__(self, bias=0):
        self.bias = bias

    def fit(self, train):
        self.fit_episodes = set(train.episode_ids)
        self.prior = float(np.mean(train.labels > 0))
        return self

    def features(self, split):
        assert self.fit_episodes.isdisjoint(set(split.episode_ids)), "base feature leaked episode"
        self.feature_calls.append((frozenset(self.fit_episodes), frozenset(split.episode_ids)))
        p = np.clip(0.25 + self.bias + 0.35 * self.prior + 0.05 * split.X[:, :10].mean(axis=1), 0.02, 0.98)
        n = split.n
        return {"p_win": p, "p_win_independent": p,
                "p_shared": np.full(n, 0.2), "p_null": np.full(n, 0.1),
                "max_pair_post": np.full(n, 0.6), "p_flip": np.full(n, 0.3),
                "since_flip": np.full(n, 12.0), "n_obs": np.sum(split.X >= 0, axis=1),
                "n_recent": np.sum(split.X[:, -8:] >= 0, axis=1),
                "machine_type": split.machine_type, "signal": split.signal}

    def predict_win(self, split):
        return self.features(split)["p_win"]


class FakeClassifier:
    def __init__(self, n_iter):
        self.n_iter = n_iter

    def fit(self, X, y):
        self.mean = float(np.mean(y))
        return self

    def predict_proba(self, X):
        p = np.clip(0.1 + 0.75 * self.mean + 0.01 * X[:, 0] + self.n_iter / 10000, 0.01, 0.99)
        return np.column_stack([1 - p, p])


class FakeMember(FakeBase):
    def predict_win(self, split):
        return np.clip(super().predict_win(split), 0.01, 0.99)


def m5_factory():
    return BoostedCorrector(base_factory=FakeBase, classifier_factory=FakeClassifier,
                            folds=4, inner_folds=3, progress=lambda _: None)


def member_factories():
    return {"m2": lambda: FakeMember(-0.06), "m3": lambda: FakeMember(0.0),
            "m4": lambda: FakeMember(0.04), "m5": m5_factory}


class CrossfitTests(unittest.TestCase):
    def test_episode_folds_hold_all_cutoffs_and_row_ids_align(self):
        train = sample_train()
        frame = pseudo_frame(train)
        self.assertEqual(len(frame.row_ids), train.n * 8)
        self.assertEqual(len(set(frame.row_ids)), len(frame.row_ids))
        for _, held in episode_folds(train, 4):
            held_eps = set(train.episode_ids[held])
            self.assertEqual(set(train.episode_ids[held]), held_eps)
            self.assertEqual(sum(ep in held_eps for ep in frame.episode_ids), len(held) * 8)
        self.assertEqual(sum(shift == 0 for _, _, shift in frame.row_ids), train.n)

    def test_cache_invalidates_same_ids_when_label_or_history_changes(self):
        train = sample_train()
        cache = FitCache()
        count = []

        class Counting(FakeBase):
            def fit(self, part):
                count.append(1)
                return super().fit(part)

        a = cache.fit("fake", train, Counting, config=(1,))
        self.assertIs(a, cache.fit("fake", train, Counting, config=(1,)))
        changed = subset_split(train, np.arange(train.n))
        changed.labels[0] = 1 - changed.labels[0]
        self.assertIsNot(a, cache.fit("fake", changed, Counting, config=(1,)))
        changed.X[0, 0] = 1 - changed.X[0, 0]
        cache.fit("fake", changed, Counting, config=(1,))
        self.assertEqual(len(count), 3)

    def test_m5_nested_features_exclude_each_feature_episode(self):
        FakeBase.feature_calls = []
        train = sample_train(8)
        model = m5_factory().fit(train)
        self.assertEqual(len(model.oof), len(pseudo_frame(train).y))
        self.assertTrue(FakeBase.feature_calls)
        self.assertTrue(all(a.isdisjoint(b) for a, b in FakeBase.feature_calls))
        self.assertIn(model.max_iter, model.ITER_GRID)
        pickle.dumps(model)  # a progress closure must not enter the checkpoint

    def test_outer_holdout_label_change_cannot_change_forecasts_or_choices(self):
        train = sample_train()
        a = Ensemble(folds=4, member_factories=member_factories(), progress=lambda _: None)
        first = a.evaluate_heldout(train, folds=2)
        changed = subset_split(train, np.arange(train.n))
        changed.labels[changed.episode_ids == "e00"] = 1 - changed.labels[changed.episode_ids == "e00"]
        b = Ensemble(folds=4, member_factories=member_factories(), progress=lambda _: None)
        second = b.evaluate_heldout(changed, folds=2)
        self.assertEqual(first["row_ids"], second["row_ids"])
        self.assertEqual(set(first["predictions"]), set(Ensemble.KEYS) | {"m6"})
        held_fold = next(i for i, f in enumerate(first["folds"]) if "e00" in f["held_episode_ids"])
        self.assertEqual(first["folds"][held_fold], second["folds"][held_fold])
        held_eps = set(first["folds"][held_fold]["held_episode_ids"])
        rows = np.isin(first["episode_ids"], list(held_eps))
        for key in first["predictions"]:
            np.testing.assert_allclose(first["predictions"][key][rows],
                                       second["predictions"][key][rows], rtol=0, atol=0)
        self.assertEqual(int(first["is_endpoint"].sum()), train.n)


if __name__ == "__main__":
    unittest.main()
