"""Focused prospective, missing-data, and machine-permutation checks."""

from dataclasses import replace
import unittest

import numpy as np
from scipy.special import expit

from slot_forecasting.adaptive import AdaptiveLagForecaster
from slot_forecasting.causal import MISSING, tick_arrays
from slot_forecasting.peer_lags import (
    PeerEvidence, PeerFeatures, PeerSettings, adaptive_trace, episode_peer_features, peer_predictions,
    prospective_evidence, selected_correction,
)
from slot_forecasting.schema import PAYOUTS, Play, Query


def manifest(spins):
    queries = tuple(Query(f"q{i}", "episode", f"m{i}", "steady", 0, spins.shape[1] - 1)
                    for i in range(len(spins)))
    history = tuple(Play(q.episode_id, q.machine_id, int(t), PAYOUTS[spins[i, t]])
                    for i, q in enumerate(queries) for t in np.flatnonzero(spins[i] != MISSING))
    return queries, history


class PeerLagsTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(79)
        self.spins = rng.integers(-1, 3, size=(4, 61), dtype=np.int8)
        self.spins[:, -1] = MISSING
        self.queries, self.history = manifest(self.spins)
        self.base = AdaptiveLagForecaster().fit(self.queries, self.history)
        self.settings = PeerSettings(2, .7)

    def test_trace_matches_truncated_reference(self):
        spins, _ = tick_arrays(self.queries, self.history)
        trace = adaptive_trace(self.queries, spins, self.base)
        for t in (0, 1, 6, 15, 40, 60):
            queries = [replace(q, cutoff=t) for q in self.queries]
            reference = self.base.predict_variants(queries, [p for p in self.history if p.tick < t])
            np.testing.assert_allclose(trace.base[:, t], [reference["adaptive_lag"][q.query_id] for q in queries],
                                       atol=1e-12, rtol=0)
            np.testing.assert_allclose(trace.no_lag_win[:, t],
                                       [1 - reference["lag_no_context"][q.query_id][0] for q in queries],
                                       atol=1e-12, rtol=0)

    def test_planted_opposite_lag_signs_and_null(self):
        rng = np.random.default_rng(1)
        horizon = 240
        spins = np.full((3, horizon + 1), MISSING, dtype=np.int8)
        spins[1:, :horizon] = rng.binomial(1, .5, (2, horizon))
        spins[0, :3] = rng.binomial(1, .5, 3)
        planted = expit(5 * (spins[1, :horizon - 3] - spins[2, :horizon - 3]))
        spins[0, 3:horizon] = rng.binomial(1, planted)
        queries, _ = manifest(spins)
        features = episode_peer_features(queries, spins, np.full(3, .5))["individual"]
        base = np.full(spins.shape, .5)
        evidence = prospective_evidence(features, spins, base)
        self.assertGreater(evidence.coefficients[0, horizon, 2], .5)
        self.assertLess(evidence.coefficients[0, horizon, 10], -.5)
        _, weights = selected_correction(features, evidence, np.array([0]), np.array([horizon]), self.settings)
        np.testing.assert_array_equal(np.flatnonzero(weights[0]), [2, 10])

        # A null relationship can still win a short local comparison by chance.
        spins[0, :horizon] = rng.binomial(1, .5, horizon)
        evidence = prospective_evidence(features, spins, base)
        ticks = np.arange(120, horizon)
        steps, _ = selected_correction(features, evidence, np.zeros(len(ticks), dtype=int), ticks, self.settings)
        self.assertLess(np.abs(steps).mean(), .12)
        self.assertLess(np.mean((spins[0, ticks] - expit(steps)) ** 2) - .25, .005)

    def test_prefix_mutation_and_focal_exclusion(self):
        t = 45
        original = adaptive_trace(self.queries, self.spins, self.base)
        changed = self.spins.copy()
        changed[:, t:-1] = (changed[:, t:-1] + 1) % 3
        updated = adaptive_trace(self.queries, changed, self.base)
        np.testing.assert_array_equal(original.base[:, :t + 1], updated.base[:, :t + 1])
        for variant, features in episode_peer_features(self.queries, self.spins, original.population_win).items():
            newer = episode_peer_features(self.queries, changed, updated.population_win)[variant]
            np.testing.assert_array_equal(features.values[:, :t + 1], newer.values[:, :t + 1])
            np.testing.assert_array_equal(features.observed[:, :t + 1], newer.observed[:, :t + 1])
            evidence = prospective_evidence(features, self.spins, 1 - original.base[..., 0])
            new_evidence = prospective_evidence(newer, changed, 1 - updated.base[..., 0])
            np.testing.assert_array_equal(evidence.coefficients[:, t], new_evidence.coefficients[:, t])
            np.testing.assert_array_equal(evidence.gain[:, t], new_evidence.gain[:, t])
        row_machines = np.arange(len(self.queries))
        row_ticks = np.full(len(row_machines), t)
        before, _ = peer_predictions(self.queries, self.spins, original, row_machines, row_ticks, (self.settings,))
        after, _ = peer_predictions(self.queries, changed, updated, row_machines, row_ticks, (self.settings,))
        for variant in before[self.settings]:
            np.testing.assert_array_equal(before[self.settings][variant], after[self.settings][variant])

        changed = self.spins.copy()
        changed[0, :t] = 1
        updated = adaptive_trace(self.queries, changed, self.base)
        self.assertGreater(abs(original.base[0, t, 0] - updated.base[0, t, 0]), .1)
        before = episode_peer_features(self.queries, self.spins, original.population_win)
        after = episode_peer_features(self.queries, changed, updated.population_win)
        for variant in before:
            np.testing.assert_array_equal(before[variant].values[0], after[variant].values[0])

    def test_permutation_ids_missing_and_ties(self):
        def forecast(queries, history):
            spins, cutoffs = tick_arrays(queries, history)
            trace = adaptive_trace(queries, spins, self.base)
            return peer_predictions(queries, spins, trace, np.arange(len(queries)), cutoffs, (self.settings,))[0][self.settings]

        original = forecast(self.queries, self.history)
        permutation = [2, 0, 3, 1]
        renamed = {q.machine_id: f"renamed{3 - i}" for i, q in enumerate(self.queries)}
        reordered = [replace(self.queries[i], machine_id=renamed[self.queries[i].machine_id]) for i in permutation]
        changed = [replace(p, machine_id=renamed[p.machine_id]) for p in reversed(self.history)]
        shuffled = forecast(reordered, changed)
        for variant in original:
            np.testing.assert_allclose(shuffled[variant], original[variant][permutation], atol=1e-14, rtol=0)

        history = [p for p in self.history if p.machine_id == self.queries[0].machine_id]
        missing = forecast(self.queries, history)
        spins, cutoffs = tick_arrays(self.queries, history)
        trace = adaptive_trace(self.queries, spins, self.base)
        for variant in missing:
            np.testing.assert_array_equal(missing[variant][0], trace.base[0, cutoffs[0]])

        shape = (1, 1, 3)
        features = PeerFeatures(np.array([[[1., 2., 3.]]]), np.ones(shape, dtype=bool), np.array([1, 2, 3]))
        evidence = PeerEvidence(np.ones(shape), np.full(shape, 20), np.full(shape, .1))
        correction, weights = selected_correction(features, evidence, np.array([0]), np.array([0]), PeerSettings(1, 1))
        np.testing.assert_allclose(weights[0], [1 / 3] * 3)
        np.testing.assert_allclose(correction, [2.])


if __name__ == "__main__":
    unittest.main()
