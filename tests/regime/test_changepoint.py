from dataclasses import replace
import json

import numpy as np

from tdx.arms.regime.changepoint import ChangeModel
from tdx.data import Split


def _split():
    y = np.full((12, 8, 256), -1, dtype=np.int8)
    y[:, :, :160] = 0
    y[6:, :, 100:160] = 1
    return Split("train", [f"episode-{i}" for i in range(12)], y,
                 np.zeros((12, 8), np.int8), np.zeros((12, 8), np.int8),
                 np.full(12, 160), np.full((12, 8), 160),
                 np.array([[f"q-{e}-{m}" for m in range(8)] for e in range(12)]),
                 condition=np.array(["stable"] * 6 + ["shared_change"] * 6))


def test_detects_shared_change_and_json_roundtrip():
    split = _split()
    model = ChangeModel().fit([split])
    ticks, fires = model.detect(split)
    np.testing.assert_array_equal(fires, np.arange(12) >= 6)
    np.testing.assert_array_equal(ticks, [-1] * 6 + [100] * 6)
    assert model.report["cv_accuracy"] == 1.0
    assert model.report["train_fire_rates"] == {"shared_change": 1.0, "stable": 0.0}
    restored = ChangeModel.from_dict(json.loads(json.dumps(model.to_dict(), allow_nan=False)))
    for expected, actual in zip(model.detect(split), restored.detect(split)):
        np.testing.assert_array_equal(expected, actual)


def test_future_and_cutoff_covariates_cannot_affect_fit_or_detect():
    split = _split()
    model = ChangeModel().fit([split])
    y = split.y.copy()
    y[:, :, 160:] = 4
    altered = replace(split, y=y, n_obs=np.full((12, 8), 256),
                      signal=np.ones((12, 8), np.int8), label=np.full((12, 8), 4),
                      target=np.ones((12, 8, 5)), reference=np.zeros((12, 8, 5)))
    assert model.to_dict() == ChangeModel().fit([altered]).to_dict()
    for expected, actual in zip(model.evidence(split), model.evidence(altered)):
        np.testing.assert_array_equal(expected, actual)


def test_missing_ticks_are_unknown_not_losing_spins():
    split = _split()
    y = np.full_like(split.y, -1)
    # Missing early data and observed class 1 late data must not become a
    # class-0-to-class-1 change. There is only one observed distribution.
    y[:, :, 100:160] = 1
    model = ChangeModel()
    _, scores = model.evidence(replace(split, y=y))
    assert np.all(scores < 0)
    assert not model.detect(replace(split, y=y))[1].any()
    assert not model.detect(replace(split, y=np.full_like(y, -1)))[1].any()


def test_fit_order_determinism_and_single_episode_cv():
    split = _split()
    model = ChangeModel().fit([split])
    ix = np.arange(11, -1, -1)
    reversed_split = replace(split, ids=list(np.array(split.ids)[ix]), y=split.y[ix],
                             condition=split.condition[ix])
    assert model.to_dict() == ChangeModel().fit([reversed_split]).to_dict()
    one = replace(split, ids=split.ids[:1], y=split.y[:1], cutoff=split.cutoff[:1],
                  condition=split.condition[:1])
    assert ChangeModel().fit([one]).report["cv_accuracy"] is None
