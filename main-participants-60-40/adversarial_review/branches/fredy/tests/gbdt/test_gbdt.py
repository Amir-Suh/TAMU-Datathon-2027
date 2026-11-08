"""Independent scalar oracle and end-to-end deterministic model checks."""
from dataclasses import replace
import numpy as np
import pytest
from tdx.data import Split, Round
from tdx.fit import dev_spec, final_spec
from tdx.arms.gbdt import build
from tdx.arms.gbdt.arm import GBDTArm
from tdx.arms.gbdt.features import features, query_features


def sample(E=3):
    rng = np.random.default_rng(17)
    y = rng.integers(0, 5, (E, 8, 96), dtype=np.int8)
    y[rng.random(y.shape) < .25] = -1
    y[..., 80:] = -1
    return Split('train', [f'episode{i}' for i in range(E)], y,
        rng.integers(0, 3, (E, 8), dtype=np.int8), np.zeros((E, 8), np.int8),
        np.full(E, 80, np.int16), (y >= 0).sum(-1), np.arange(E*8).reshape(E, 8).astype(object),
        label=rng.integers(0, 5, (E, 8), dtype=np.int8))


def slow(y, mtype, e, m, t):
    hist = y[e, m, :t]
    observed = hist[hist >= 0]
    lag = [int(hist[-k]) if t >= k else -1 for k in range(1, 9)]
    out = lag + [5*lag[0]+lag[1] if min(lag[:2]) >= 0 else -1]
    for half in (4, 12, 24, 64):
        weights = np.exp2(-np.arange(len(observed)-1, -1, -1)/half)
        out.extend([np.sum(weights * (observed == c))/weights.sum() if len(observed) else -1 for c in range(5)])
    out.extend(np.sum(observed[-n:] > 0) for n in (5, 20, 50))
    out.extend([len(observed), mtype[e, m]])
    for n in (1, 5, 10):
        if n == 1:
            z = np.asarray([h[h >= 0][-1] if np.any(h >= 0) else -1
                            for j, h in enumerate(y[e, :, :t]) if j != m])
        else:
            z = np.delete(y[e, :, max(0, t-n):t], m, axis=0).ravel()
        z = z[z >= 0]
        out.extend([np.mean(z > 0) if len(z) else -1,
                    np.mean(np.asarray([0, 1, 2, 5, 20])[z]) if len(z) else -1,
                    np.max(z) if len(z) else -1])
    rates, episode, valid, epvalid = [], [], [], []
    for n in (15, 80):
        a = hist[max(0, t-n):]; a = a[a >= 0]
        valid.append(len(a) > 0)
        rates.append(np.mean(a > 0) if len(a) else 0)
        a = y[e, :, max(0, t-n):t]; a = a[a >= 0]
        epvalid.append(len(a) > 0)
        episode.append(np.mean(a > 0) if len(a) else 0)
    return np.asarray(out + [rates[0]-rates[1] if all(valid) else np.nan, episode[0]-episode[1] if all(epvalid) else np.nan, t/256.], np.float32)


def test_three_episode_scalar_reference():
    s = sample()
    x, names, groups = features(s.y, s.mtype)
    assert x.shape[-1] == len(names) == len(groups) == 46
    # All rows of three episodes, including empty prefix and next tick.
    for e in range(3):
        for m in range(8):
            for t in range(97):
                np.testing.assert_allclose(x[e, m, t], slow(s.y, s.mtype, e, m, t), atol=1e-6, rtol=1e-6)


def test_feature_future_perturbation():
    s = sample()
    x, _, _ = features(s.y, s.mtype)
    dirty = s.y.copy(); dirty[..., 31:] = 4
    other, _, _ = features(dirty, s.mtype)
    np.testing.assert_array_equal(x[..., :32, :], other[..., :32, :])
    a, _, _ = query_features(s)
    dirty = s.y.copy(); dirty[..., 80:] = 4
    b, _, _ = query_features(replace(s, y=dirty, n_obs=np.full((3, 8), 999)))
    np.testing.assert_array_equal(a, b)


def test_fit_determinism_and_save_load(tmp_path):
    train = sample(10)
    val = replace(sample(), name='validation', label=None)
    rnd = Round('test', tmp_path, {'train': train, 'validation': val, 'test': replace(val, name='test')}, 60.)
    params = dict(rounds=10, patience=3)
    a = build(**params).fit(rnd, dev_spec())
    b = build(**params).fit(rnd, dev_spec())
    pred = a.predict(val)
    np.testing.assert_array_equal(pred, b.predict(val))
    a.save(tmp_path/'model')
    np.testing.assert_array_equal(pred, GBDTArm.load(tmp_path/'model').predict(val))
    assert pred.dtype == np.float64 and np.isfinite(pred).all()
    assert np.all((pred > 0) & (pred < 1))
    np.testing.assert_allclose(pred.sum(-1), 1., atol=1e-12, rtol=0)


def test_final_soft_rows(tmp_path):
    train = sample(10)
    s = sample()
    val = replace(s, name='validation', target=np.full((3, 8, 5), .2))
    rnd = Round('test', tmp_path, {'train': train, 'validation': val, 'test': replace(val, name='test', label=None, target=None)}, 60.)
    a = build(rounds=2).fit(rnd, final_spec())
    assert a.fit_rows > 24 * 5
    assert a.predict(val).shape == (3, 8, 5)


def test_sparse_divergence_requires_observations_in_both_windows():
    y = np.full((1, 8, 96), -1, np.int8)
    y[0, 0, 1] = 4
    y[0, 1, 90] = 0
    x, names, _ = features(y, np.zeros((1, 8), np.int8))
    self_div = names.index('self_divergence')
    episode_div = names.index('episode_divergence')
    # The old win remains in the 80-tick window, but an empty recent
    # window cannot be interpreted as an observed loss rate of zero.
    assert np.isnan(x[0, 0, 40, self_div])
    assert np.isnan(x[0, 0, 40, episode_div])
    # An observed recent loss supplies a real zero rate for the episode.
    assert x[0, 0, 91, episode_div] == 0.
    assert np.isnan(x[0, 0, 91, self_div])
    assert x[0, 1, 91, self_div] == 0.


def test_neighbour_one_uses_each_other_machines_last_observation():
    y = np.full((1, 8, 32), -1, np.int8)
    y[0, 0, 19] = 4  # Excluded from machine zero's neighbour summaries.
    y[0, 1, 1] = 3
    y[0, 2, 10] = 0
    y[0, 3, 25] = 4  # A future observation must not be carried backward.
    x, names, _ = features(y, np.zeros((1, 8), np.int8))
    got = x[0, 0, 20]
    assert got[names.index('other1_win')] == .5
    assert got[names.index('other1_payout')] == 2.5
    assert got[names.index('other1_maxclass')] == 3.
    # The 5/10-tick windows still use calendar observations, not carry-forward.
    assert got[names.index('other5_win')] == -1.
    assert got[names.index('other10_win')] == 0.
    assert got[names.index('other10_payout')] == 0.
    assert got[names.index('other10_maxclass')] == 0.
    assert x[0, 0, 0, names.index('other1_win')] == -1.


def test_final_rows_and_soft_target_weights_exact(tmp_path, monkeypatch):
    import tdx.arms.gbdt.arm as implementation

    train = sample(10)
    s = sample()
    target = np.broadcast_to(np.array([.05, .1, .15, .3, .4]), (3, 8, 5)).copy()
    labels = s.label.copy()
    labels[0, 0] = -1
    val = replace(s, name='validation', label=labels, target=target)
    test = replace(s, name='test', label=None, target=None)
    rnd = Round('test', tmp_path, {'train': train, 'validation': val, 'test': test}, 60.)
    fitted = []

    class Booster:
        best_iteration = 1

    def capture(params, data, **kwargs):
        fitted.append((data.data.copy(), np.asarray(data.label).copy(), np.asarray(data.weight).copy()))
        return Booster()

    monkeypatch.setattr(implementation.lgb, 'train', capture)
    arm = build(rounds=2, w_cut=7.).fit(rnd, final_spec())
    _, y, weight = fitted[-1]
    tick_count = 0
    for split in (train, val, test):
        for hist in split.y.reshape(-1, split.y.shape[-1]):
            tick_count += max(0, np.count_nonzero(hist[:80] >= 0) - 8)
    hard_count = train.label.size + np.count_nonzero(labels >= 0)
    assert arm.fit_rows == tick_count + hard_count + target.size
    np.testing.assert_array_equal(weight[:tick_count], np.ones(tick_count))
    np.testing.assert_array_equal(weight[tick_count:tick_count + hard_count], np.full(hard_count, 7.))
    np.testing.assert_array_equal(weight[-target.size:], 7. * target.ravel())
    np.testing.assert_array_equal(y[-target.size:], np.tile(np.arange(5), target.shape[0] * 8))
    np.testing.assert_allclose(weight[-target.size:].reshape(-1, 5).sum(-1), 7., rtol=0, atol=1e-14)


def test_final_fit_reproducible_across_hash_seeds(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys

    script = '''
import runpy, sys
from dataclasses import replace
from pathlib import Path
import numpy as np
from tdx.data import Round
from tdx.fit import final_spec
from tdx.arms.gbdt import build
sample = runpy.run_path(sys.argv[1])['sample']
train, s = sample(10), sample()
target = np.broadcast_to(np.array([.05, .1, .15, .3, .4]), (3, 8, 5)).copy()
val = replace(s, name='validation', target=target)
rnd = Round('test', Path(sys.argv[2]), {'train': train, 'validation': val,
    'test': replace(val, name='test', label=None, target=None)}, 60.)
arm = build(rounds=5, patience=2, threads=1).fit(rnd, final_spec())
np.save(sys.argv[2], arm.predict(val))
'''
    paths = []
    for seed in ('1', '8675309'):
        path = tmp_path / f'prediction_{seed}.npy'
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        subprocess.run([sys.executable, '-c', script, str(Path(__file__).resolve()), str(path)],
                       check=True, env=env, capture_output=True, text=True, timeout=60)
        paths.append(path)
    first, second = (np.load(path) for path in paths)
    assert np.max(np.abs(first - second)) == 0.0
