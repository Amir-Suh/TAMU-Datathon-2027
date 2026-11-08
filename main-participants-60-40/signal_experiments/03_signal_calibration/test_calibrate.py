import unittest

import numpy as np

from calibrate import Cutoffs, fit_offsets, fit_variants, reweight_win


class CalibrationTest(unittest.TestCase):
    def test_adjustment_preserves_conditional_payout_mix(self):
        raw = np.array([[0.6, 0.1, 0.15, 0.05, 0.1], [0.1, 0.3, 0.2, 0.2, 0.2]])
        updated = reweight_win(raw, np.array([1, -1]), (0.0, 0.7))
        np.testing.assert_allclose(updated.sum(axis=1), 1)
        np.testing.assert_allclose(
            updated[:, 1:] / updated[:, 1:].sum(axis=1, keepdims=True),
            raw[:, 1:] / raw[:, 1:].sum(axis=1, keepdims=True),
        )
        self.assertLess(updated[0, 0], raw[0, 0])
        self.assertGreater(updated[1, 0], raw[1, 0])

    def test_signal_fit_finds_positive_slope(self):
        signal = np.tile(np.array([-1, 1]), 10)
        p = np.full(len(signal), 0.5)
        outcomes = np.maximum(signal, 0)
        episodes = np.repeat(np.arange(10), 2)
        intercept, slope = fit_offsets(p, signal, outcomes, episodes, with_signal=True)
        self.assertAlmostEqual(intercept, 0)
        self.assertGreater(slope, 0)

    def test_held_episode_label_cannot_change_its_forecast(self):
        episodes = np.repeat(np.arange(4), 2)
        raw = np.tile(np.array([0.5, 0.2, 0.2, 0.05, 0.05]), (len(episodes), 1))
        signal = np.tile([-1, 1], 4)
        outcomes = (signal > 0).astype(float)
        changed_outcomes = outcomes.copy()
        changed_outcomes[episodes == 0] = 1 - changed_outcomes[episodes == 0]
        before = Cutoffs(tuple(range(8)), episodes, signal, outcomes)
        after = Cutoffs(tuple(range(8)), episodes, signal, changed_outcomes)
        first, _ = fit_variants(before, raw, folds=episodes)
        second, _ = fit_variants(after, raw, folds=episodes)
        for model in first:
            np.testing.assert_array_equal(first[model][:2], second[model][:2])
        self.assertFalse(np.array_equal(first["intercept_signal"][2:], second["intercept_signal"][2:]))


if __name__ == "__main__":
    unittest.main()
