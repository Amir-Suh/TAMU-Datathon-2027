from dataclasses import replace

import numpy as np
import pytest
from tdx.data import load_round
from tdx.fit import dev_spec
from tdx.run import fit_round, arm_module, build_arm
from tdx.backtest import truncate, make_pseudo_split, tail_logloss


def test_truncate():
    rnd = load_round('data/main', answers=False)
    hold = 30
    rt, full = truncate(rnd, hold)

    for name, s_trunc in rt.splits.items():
        s_full = full[name]
        assert np.array_equal(s_trunc.cutoff, s_full.cutoff - hold)

        for e in range(len(s_trunc.cutoff)):
            new_co = int(s_trunc.cutoff[e])
            assert np.all(s_trunc.y[e, :, new_co:] == -1)
            assert np.array_equal(s_trunc.y[e, :, :new_co], s_full.y[e, :, :new_co])

        assert np.all(s_trunc.signal == 0)
        np.testing.assert_array_equal(s_trunc.n_obs, (s_trunc.y >= 0).sum(axis=-1))
        assert s_trunc.label is None
        assert s_trunc.condition is None
        assert s_trunc.target is None
        assert s_trunc.reference is None

        assert s_full.cutoff is rnd.splits[name].cutoff
        assert s_full.y is rnd.splits[name].y


def test_no_leak():
    rnd = load_round('data/main', answers=False)
    rt, full = truncate(rnd, 30)
    module = arm_module('pooled')
    arm = build_arm(module, module.VARIANTS['pooled_recent'])
    spec = dev_spec(False)
    fitted_rt = fit_round(rt, spec)
    splits = dict(fitted_rt.splits)
    if splits['train'].label is None:
        splits['train'] = replace(splits['train'], label=np.full(splits['train'].query_ids.shape, -1, dtype=np.int8))
    arm.fit(replace(fitted_rt, splits=splits), spec)

    val_trunc = rt.splits['validation']
    cutoff_trunc = val_trunc.cutoff

    # Pseudo-query forecasts for j = 0..4 on clean full['validation']
    clean_preds = []
    for j in range(5):
        split_j = make_pseudo_split(val_trunc, full['validation'], j)
        clean_preds.append(arm.predict(split_j))

    # Corrupt full['validation'].y at ticks >= cutoff_trunc + 5 (set to 4)
    corrupted_y = full['validation'].y.copy()
    for e in range(len(cutoff_trunc)):
        corrupted_y[e, :, int(cutoff_trunc[e]) + 5:] = 4
    corrupted_val = replace(full['validation'], y=corrupted_y)

    for j in range(5):
        split_corrupt = make_pseudo_split(val_trunc, corrupted_val, j)
        pred_corrupt = arm.predict(split_corrupt)
        np.testing.assert_array_equal(pred_corrupt, clean_preds[j])


def test_sanity():
    res_recent = tail_logloss('pooled', 'pooled_recent')
    res_pooled = tail_logloss('pooled', 'pooled')

    assert np.isfinite(res_recent)
    assert 0.5 <= res_recent <= 2.0
    assert res_recent.n > 0
    assert np.isfinite(res_recent.first)
    assert np.isfinite(res_recent.mid)
    assert np.isfinite(res_recent.last)

    assert np.isfinite(res_pooled)
    assert 0.5 <= res_pooled <= 2.0
    assert res_pooled.n > 0
    assert np.isfinite(res_pooled.first)
    assert np.isfinite(res_pooled.mid)
    assert np.isfinite(res_pooled.last)


def test_later_query_signal_is_unavailable_in_pseudo_queries():
    rnd = load_round('data/main', answers=False)
    rt, full = truncate(rnd, 30)
    val = full['validation']
    truncated = rt.splits['validation']
    perturbed = replace(val, signal=np.where(val.signal == 1, -1, 1).astype(val.signal.dtype))
    assert np.any(perturbed.signal != val.signal)
    for offset in (0, 29):
        pseudo = make_pseudo_split(truncated, val, offset)
        dirty = make_pseudo_split(replace(truncated, signal=perturbed.signal), perturbed, offset)
        np.testing.assert_array_equal(pseudo.signal, np.zeros_like(val.signal))
        np.testing.assert_array_equal(pseudo.signal, dirty.signal)
    actual = make_pseudo_split(truncated, val, 30)
    np.testing.assert_array_equal(actual.signal, val.signal)


def test_tail_report_rejects_forecasts_from_before_signal_fix(tmp_path):
    from tdx.tools.tail_eval import load_tail_dataset

    path = tmp_path / 'tail_m3_7_h30.npz'
    np.savez(path, preds=np.full((1, 1, 8, 5), .2), outs=np.zeros((1, 1, 8), dtype=int))
    with pytest.raises(ValueError, match='neutral pseudo-cutoff signal'):
        load_tail_dataset(tmp_path, ['m3'], [30])


def test_m3_win_backtest_masks_the_later_signal():
    from tdx.arms.m3flip.amir.common.data import Split, truncate as truncate_m3

    train = Split('train', ['q'], np.array(['e']), ['m'], np.array([0]),
                  np.array([1]), np.array([3]), np.array([[0, 1, 2, -1]]),
                  labels=np.array([3]))
    earlier = truncate_m3(train, 1)
    np.testing.assert_array_equal(earlier.signal, [0])
    np.testing.assert_array_equal(earlier.X, [[0, 1, -1]])
    np.testing.assert_array_equal(earlier.labels, [2])
    actual = truncate_m3(train, 0)
    np.testing.assert_array_equal(actual.signal, train.signal)
    np.testing.assert_array_equal(actual.labels, train.labels)


def test_m3_payout_backtest_uses_actual_signal_only_at_the_query(monkeypatch):
    from tdx.arms.m3flip.amir.common import components
    from tdx.arms.m3flip.amir.common.data import N_CONTEXTS, Split

    train = Split('train', ['q'], np.array(['e']), ['m'], np.array([0]),
                  np.array([1]), np.array([2]), np.array([[1, 0, -1]]),
                  labels=np.array([2]))
    priors = np.tile([.25, .25, .25, .25], (N_CONTEXTS, 1))
    priors[train.context] = [.1, .7, .1, .1]
    strength = 5.0
    monkeypatch.setattr(components.SizeComponent, 'GRID', [{'strength': strength, 'half_life': np.inf}])
    original_loss = components.categorical_logloss

    def check_score(pred, cat, mask):
        np.testing.assert_array_equal(mask, [[True, False, True]])
        np.testing.assert_allclose(pred[0, 0], priors[1])
        np.testing.assert_allclose(
            pred[0, 2], (strength * priors[2] + np.array([1, 0, 0, 0])) / (strength + 1))
        return original_loss(pred, cat, mask)

    monkeypatch.setattr(components, 'categorical_logloss', check_score)
    components.SizeComponent().fit(train, priors)
