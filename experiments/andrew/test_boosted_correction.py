"""Verify the supplied offset, weighted Newton update, and nested episode isolation."""

import unittest

import numpy as np
from scipy.special import expit, logit

from run_boosted_correction import combine_samples, cross_fitted_inputs, fit_corrections
from slot_forecasting.adaptive import AdaptiveLagForecaster
from slot_forecasting.boosted_correction import (
    BoostInputs, OffsetCalibration, OffsetTrees, build_inputs, corrected_probabilities,
)
from slot_forecasting.peer_lags import adjust_win
from slot_forecasting.schema import Play, Query
from test_peer_lags import manifest


class BoostedCorrectionTest(unittest.TestCase):
    def test_weighted_newton_leaf_uses_and_preserves_offset(self):
        probability = np.array([.1, .25, .6, .8])
        offset = logit(probability)
        outcomes = np.array([0, 1, 1, 0])
        features = np.ones((len(outcomes), 1))
        weights = np.array([2., 1., 3., 2.])
        curvature = probability * (1 - probability)
        expected = np.sum(weights * (outcomes - probability)) / np.sum(weights * curvature)
        trees = OffsetTrees(rounds=1, min_samples_leaf=1, shrinkage=.05).fit(features, offset, outcomes, weights)
        np.testing.assert_allclose(trees.trees[0].predict(features), expected, atol=1e-12, rtol=0)
        np.testing.assert_allclose(trees.corrections(features, (1,))[1], .05 * expected, atol=1e-12, rtol=0)

        sizes = np.array([.45, .30, .20, .05])
        base = np.column_stack([1 - probability, probability[:, None] * sizes])
        calibration = OffsetCalibration()
        calibration.coefficients = np.zeros(2)
        results = corrected_probabilities(BoostInputs(features, offset, base), calibration, trees, budgets=(1,))
        np.testing.assert_array_equal(adjust_win(base, np.zeros(len(base))), base)
        np.testing.assert_array_equal(results["zero_correction"], base)
        np.testing.assert_array_equal(results["calibration"], base)
        corrected = expit(offset + .05 * expected)
        np.testing.assert_allclose(results["trees_1"][:, 0], 1 - corrected, atol=1e-12, rtol=0)
        np.testing.assert_allclose(results["trees_1"][:, 1:], corrected[:, None] * sizes, atol=1e-12, rtol=0)

        extreme = OffsetTrees(rounds=1, min_samples_leaf=1, shrinkage=.05).fit(
            features, np.full(len(offset), -20.), np.ones(len(offset)), weights)
        np.testing.assert_array_equal(extreme.corrections(features, (1,))[1], np.full(len(offset), .05))

    def test_at_or_after_tick_changes_leave_features_fixed(self):
        rng = np.random.default_rng(31)
        spins = rng.integers(-1, 3, size=(4, 61), dtype=np.int8)
        spins[:, -1] = -1
        queries, history = manifest(spins)
        model = AdaptiveLagForecaster().fit(queries, history)
        machines = np.arange(len(queries))
        tick = 37
        ticks = np.full(len(queries), tick)
        original = build_inputs(queries, spins, machines, ticks, model)
        changed = spins.copy()
        changed[:, tick:] = (spins[:, tick:] + 1) % 4
        updated = build_inputs(queries, changed, machines, ticks, model)
        for field in ("base", "offset", "features"):
            np.testing.assert_array_equal(getattr(original, field), getattr(updated, field))

    def test_outer_episode_labels_cannot_change_base_or_correction_fit(self):
        rng = np.random.default_rng(55)
        queries = tuple(Query(f"q{e}_{m}", f"e{e}", f"m{m}", "steady", m, 35)
                        for e in range(8) for m in range(2))
        history = tuple(Play(q.episode_id, q.machine_id, t, int(rng.integers(0, 3)))
                        for q in queries for t in range(q.cutoff))
        labels = {q.query_id: 0 for q in queries}
        folds = [{f"e{i}", f"e{i + 4}"} for i in range(4)]
        original = cross_fitted_inputs(queries, history, labels, folds)
        changed_labels = {q.query_id: 5 if q.episode_id in folds[0] else labels[q.query_id] for q in queries}
        changed = cross_fitted_inputs(queries, history, changed_labels, folds)

        np.testing.assert_array_equal(original[1][0].base, changed[1][0].base)
        np.testing.assert_array_equal(original[1][0].features, changed[1][0].features)
        self.assertGreater(np.max(np.abs(original[1][1].base - changed[1][1].base)), 0)
        coefficients = []
        for samples, _, inner in (original, changed):
            training = combine_samples(samples[1:], [inner[0, i] for i in (1, 2, 3)])
            calibration, _ = fit_corrections(*training, rounds=0)
            coefficients.append(calibration.coefficients)
        np.testing.assert_array_equal(coefficients[0], coefficients[1])


if __name__ == "__main__":
    unittest.main()
