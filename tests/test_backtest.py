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
        from dataclasses import replace
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
    from dataclasses import replace
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
