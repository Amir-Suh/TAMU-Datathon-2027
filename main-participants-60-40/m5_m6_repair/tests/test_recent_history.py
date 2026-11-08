"""Regression checks for per-query history features used by M5 and M6."""
from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.common.data import MISSING, N_CONTEXTS, Split
from models.m3_lag_pair_flip.model import FlipLagPairModel
from models.m4_lag_pair_flip_shared.model import SharedFlipLagPairModel


def make_split() -> Split:
    cutoffs = np.array([40, 40, 40, 104, 104])
    X = np.full((5, 105), MISSING, dtype=np.int8)
    for row, cutoff in enumerate(cutoffs):
        for tick in range(cutoff):
            if (tick + 3 * row) % 11:
                X[row, tick] = (tick + row) % 5
    return Split(
        name="history-feature-test",
        query_ids=[f"q{i}" for i in range(5)],
        episode_ids=np.array(["early"] * 3 + ["late"] * 2),
        machine_ids=[f"m{i}" for i in range(5)],
        machine_type=np.array([0, 1, 2, 1, 0]),
        signal=np.array([-1, 0, 1, 1, -1]),
        cutoff=cutoffs,
        X=X,
    )


def take_rows(split: Split, rows: list[int], width: int | None = None) -> Split:
    return replace(
        split,
        query_ids=[split.query_ids[i] for i in rows],
        episode_ids=split.episode_ids[rows],
        machine_ids=[split.machine_ids[i] for i in rows],
        machine_type=split.machine_type[rows],
        signal=split.signal[rows],
        cutoff=split.cutoff[rows],
        X=split.X[rows, :width].copy(),
    )


def configured(model_class: type[FlipLagPairModel]) -> FlipLagPairModel:
    """Fixed parameters avoid training and make forecast regressions reproducible."""
    model = model_class(max_lag=4, tau_step=4, min_segment=8, max_tick=108)
    model.win_prior = np.full(N_CONTEXTS, 0.4)
    model.hi = np.full(N_CONTEXTS, 0.85)
    model.lo = np.full(N_CONTEXTS, 0.08)
    model.pair_prior = np.full(len(model.pairs), 1 / len(model.pairs))
    model.tau_q = np.full(len(model.tau_grid), 1 / len(model.tau_grid))
    model.params = {"kappa": 20.0}
    model.size.prior = np.tile([0.5, 0.3, 0.15, 0.05], (N_CONTEXTS, 1))
    model.size.params = {"strength": 100.0, "half_life": float("inf")}
    return model


def scalar_counts(split: Split) -> tuple[np.ndarray, np.ndarray]:
    """Independent tick-by-tick oracle, including missing payouts and zero wins."""
    observed, recent = [], []
    for row, cutoff in enumerate(split.cutoff):
        ticks = [t for t in range(int(cutoff)) if split.X[row, t] != MISSING]
        observed.append(len(ticks))
        recent.append(sum(t >= cutoff - 64 for t in ticks))
    return np.asarray(observed, dtype=float), np.asarray(recent, dtype=float)


class RecentHistoryFeatureTest(unittest.TestCase):
    def test_history_counts_match_scalar_oracle_at_each_cutoff(self) -> None:
        split = make_split()
        expected_obs, expected_recent = scalar_counts(split)
        for model_class in (FlipLagPairModel, SharedFlipLagPairModel):
            with self.subTest(model=model_class.__name__):
                features = configured(model_class).features(split)
                np.testing.assert_array_equal(features["n_obs"], expected_obs)
                np.testing.assert_array_equal(features["n_recent"], expected_recent)
                self.assertEqual(features["n_recent"][0], features["n_obs"][0])
                self.assertLess(features["n_recent"][3], features["n_obs"][3])

    def test_padding_and_unrelated_episode_leave_all_features_unchanged(self) -> None:
        full = make_split()
        early = take_rows(full, [0, 1, 2], width=41)
        order = [2, 0, 4, 1, 3]
        reordered = take_rows(full, order)
        for model_class in (FlipLagPairModel, SharedFlipLagPairModel):
            with self.subTest(model=model_class.__name__):
                model = configured(model_class)
                early_features = model.features(early)
                full_features = model.features(full)
                ordered_features = model.features(reordered)
                for key in early_features:
                    np.testing.assert_allclose(early_features[key], full_features[key][:3], rtol=0, atol=1e-13,
                                               err_msg=f"{key} changed when a later episode was added")
                    np.testing.assert_allclose(full_features[key][order], ordered_features[key], rtol=0, atol=1e-13,
                                               err_msg=f"{key} changed with row order")

    def test_counts_ignore_values_at_cutoff_and_in_future_storage(self) -> None:
        split = make_split()
        changed = replace(split, X=split.X.copy())
        changed.X[0, split.cutoff[0]:] = 4
        changed.X[3, split.cutoff[3]] = 0
        for model_class in (FlipLagPairModel, SharedFlipLagPairModel):
            with self.subTest(model=model_class.__name__):
                model = configured(model_class)
                original = model.features(split)
                mutated = model.features(changed)
                for key in ("n_obs", "n_recent"):
                    np.testing.assert_array_equal(mutated[key], original[key])
        # Other posterior features assume the sanitized Split guaranteed by load_split.

    def test_forecasts_are_unchanged_by_the_feature_fix(self) -> None:
        split = make_split()
        # Baselines generated from the pre-fix M3/M4 code with these fixed parameters.
        expected = {
            FlipLagPairModel: [
                [0.4119902683425104, 0.2598182535230768, 0.16865395404129546, 0.10028072942995947, 0.059256794663157863],
                [0.34727974924977745, 0.28619272532894374, 0.19079515021929583, 0.1154812751327317, 0.060251100069251315],
                [0.37190087459897936, 0.28240115715704805, 0.18015246232432375, 0.10711768030094926, 0.058427825618699594],
                [0.266024758179949, 0.2893959524890487, 0.2013189234706426, 0.14260090412503848, 0.1006594617353213],
                [0.2711483150132643, 0.28574299013684523, 0.20291893502471617, 0.1408008936906194, 0.09938886613455486],
            ],
            SharedFlipLagPairModel: [
                [0.41367979928803905, 0.25907171659365724, 0.16816935989412837, 0.09999259236948173, 0.05908653185469375],
                [0.34780689706106716, 0.28596159128860904, 0.19064106085907268, 0.11538801051996504, 0.060202440271286115],
                [0.37287524803361083, 0.28196306677558586, 0.17987299087408062, 0.10695150808729117, 0.05833718622943155],
                [0.2659081339897229, 0.289441935741195, 0.20135091181996173, 0.14262356253913955, 0.10067545590998087],
                [0.27116838946142696, 0.28573512004069057, 0.2029133461158527, 0.14079701567222433, 0.09938612870980541],
            ],
        }
        for model_class, values in expected.items():
            with self.subTest(model=model_class.__name__):
                model = configured(model_class)
                np.testing.assert_allclose(model.predict(split), values, rtol=0, atol=1e-12)
                np.testing.assert_allclose(model.predict_win(split), 1 - np.asarray(values)[:, 0], rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
